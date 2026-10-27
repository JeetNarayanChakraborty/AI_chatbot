import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pymupdf4llm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy

load_dotenv()


# SETUP: Generation Model (70B) vs Judge Model (8B)

groq_gen_llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.0, api_key=os.getenv("GROQ_API_KEY")
)

# 8B Model for judging/grading (Max efficiency, avoids rate limits)
groq_judge_llm = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0.0, api_key=os.getenv("GROQ_API_KEY")
)

# Gemini for embeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")
)

ragas_llm = LangchainLLMWrapper(groq_judge_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)


# LOAD OR BUILD EVALUATION VECTOR STORE

EVAL_INDEX_PATH = "faiss_index_eval"
EVAL_PARENT_META = "parent_lookup_eval.json"


def build_eval_vector_store():
    print("Building new vector store for the Golden Dataset...")
    md_text = pymupdf4llm.to_markdown("data/eval_document.pdf")
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    )
    parent_chunks = markdown_splitter.split_text(md_text)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    final_children = []
    parent_lookup = {}

    for i, parent in enumerate(parent_chunks):
        parent_id = f"parent_eval_{i}"
        parent_lookup[parent_id] = parent.page_content
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
            final_children.append(child)

    with open(EVAL_PARENT_META, "w") as f:
        json.dump(parent_lookup, f)

    vector_store = FAISS.from_documents(final_children, gemini_embeddings)
    vector_store.save_local(EVAL_INDEX_PATH)
    return vector_store, parent_lookup, final_children


if os.path.exists(EVAL_INDEX_PATH):
    print(f"Loading existing Golden Dataset index...")
    vector_store = FAISS.load_local(
        EVAL_INDEX_PATH, gemini_embeddings, allow_dangerous_deserialization=True
    )
    with open(EVAL_PARENT_META, "r") as f:
        parent_lookup = json.load(f)

    # If we need to generate the dataset, we need the children too
    if not os.path.exists("ragas_testset.json"):
        _, _, final_children = build_eval_vector_store()
    else:
        final_children = []
else:
    vector_store, parent_lookup, final_children = build_eval_vector_store()

retriever = vector_store.as_retriever(search_kwargs={"k": 4})


def retrieve_with_parents(query):
    child_docs = retriever.invoke(query)
    parent_docs = []
    seen = set()
    for doc in child_docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id and parent_id not in seen:
            parent_content = parent_lookup.get(parent_id, doc.page_content)
            parent_docs.append(parent_content)
            seen.add(parent_id)
    return parent_docs


# GENERATE OR LOAD DATASET

if os.path.exists("ragas_testset.json"):
    print("Found existing ragas_testset.json. Loading...")
    testset_df = pd.read_json("ragas_testset.json")
else:
    print("Generating new 30-question dataset...")
    generator = TestsetGenerator.from_langchain(
        llm=ragas_llm, embedding_model=ragas_embeddings
    )

    # Define the new synthesizer distribution
    # SingleHop = Simple questions (50%), MultiHop = Complex/Multi-context questions (50%)
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=ragas_llm), 0.5),
    ]

    testset = generator.generate_with_langchain_docs(
        documents=final_children,  # Fixed: uses final_children instead of dummy_docs
        testset_size=30,
        query_distribution=query_distribution,
    )
    testset_df = testset.to_pandas()
    testset_df.to_json("ragas_testset.json", orient="records", indent=2)


questions = testset_df["question"].tolist()
references = testset_df["ground_truth"].tolist()


# EVALUATION HELPER


def run_ragas_eval(answers, contexts, label):
    print(f"Evaluating {label} (using 8B judge + 2 metrics)...")
    samples = [
        SingleTurnSample(user_input=q, response=a, retrieved_contexts=c, reference=r)
        for q, a, c, r in zip(questions, answers, contexts, references)
    ]
    dataset = EvaluationDataset(samples=samples)

    # ONLY 2 METRICS
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    result = evaluate(dataset=dataset, metrics=metrics)
    df = result.to_pandas()
    scores = {
        col: round(float(df[col].mean()), 4)
        for col in df.columns
        if col not in ["user_input", "response", "retrieved_contexts", "reference"]
    }

    print(f"--- {label} Scores ---")
    for k, v in scores.items():
        print(f"{k}: {v}")
    return scores


# RUN BASELINES

print("\nRunning Single-Shot Baseline...")
ss_answers, ss_contexts = [], []
for q in questions:
    ctx_list = retrieve_with_parents(q)
    ctx = "\n\n".join(ctx_list)
    ans = groq_gen_llm.invoke(
        f"Answer using ONLY this context:\n{ctx}\n\nQuestion: {q}"
    ).content
    ss_answers.append(ans)
    ss_contexts.append(ctx_list)
baseline_single = run_ragas_eval(ss_answers, ss_contexts, "Single-Shot")

print("\nRunning Agentic ReAct Baseline (Simplified)...")
react_answers, react_contexts = [], []
for q in questions:
    ctx_list = retrieve_with_parents(q)
    ctx = "\n\n".join(ctx_list)

    # Grade
    grade = groq_gen_llm.invoke(
        f"Can you answer '{q}' using ONLY this context? Reply YES or NO.\nContext:\n{ctx}"
    ).content
    if "NO" in grade.upper():
        refined_q = groq_gen_llm.invoke(
            f"Original question: {q}. Generate ONE new search query to find missing info."
        ).content
        new_ctx_list = retrieve_with_parents(refined_q)
        ctx_list.extend(new_ctx_list)
        ctx = "\n\n".join(ctx_list)

    ans = groq_gen_llm.invoke(
        f"Answer using ONLY this context:\n{ctx}\n\nQuestion: {q}"
    ).content
    react_answers.append(ans)
    react_contexts.append(ctx_list)
baseline_react = run_ragas_eval(react_answers, react_contexts, "ReAct")


# DSPy OPTIMIZATION

print("\nConfiguring DSPy (70B)...")
dspy_lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
dspy.configure(lm=dspy_lm)


class RAGGenerator(dspy.Signature):
    """Answer the question using only the provided context."""

    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()


class RAGPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(RAGGenerator)

    def forward(self, question, context):
        return self.generator(question=question, context=context)


def rag_metric(example, pred, trace=None):
    return example.ground_truth.lower() in pred.answer.lower()


trainset = []

for _, row in testset_df.iterrows():
    ctx = "\n\n".join(retrieve_with_parents(row["question"]))
    trainset.append(
        dspy.Example(
            question=row["question"], context=ctx, ground_truth=row["ground_truth"]
        ).with_inputs("question", "context")
    )

# BFS Optimization

print("\nRunning BootstrapFewShot...")
student = RAGPipeline()
optimizer_bfs = BootstrapFewShot(
    metric=rag_metric, max_bootstrapped_demos=3, max_labeled_demos=3
)
optimized_bfs = optimizer_bfs.compile(student, trainset=trainset)
optimized_bfs.save("optimized_bfs.json")


# BFS Eval

bfs_answers, bfs_contexts = [], []
for q in questions:
    ctx_list = retrieve_with_parents(q)
    ctx = "\n\n".join(ctx_list)
    ans = optimized_bfs(question=q, context=ctx).answer
    bfs_answers.append(ans)
    bfs_contexts.append(ctx_list)
bfs_scores = run_ragas_eval(bfs_answers, bfs_contexts, "BootstrapFewShot")


# MIPROv2 Optimization (2 trials)

print("\nRunning MIPROv2 (num_trials=2)...")
optimizer_mipro = MIPROv2(metric=rag_metric, num_trials=2)
optimized_mipro = optimizer_mipro.compile(RAGPipeline(), trainset=trainset)
optimized_mipro.save("optimized_mipro.json")


# MIPROv2 Eval
mipro_answers, mipro_contexts = [], []
for q in questions:
    ctx_list = retrieve_with_parents(q)
    ctx = "\n\n".join(ctx_list)
    ans = optimized_mipro(question=q, context=ctx).answer
    mipro_answers.append(ans)
    mipro_contexts.append(ctx_list)
mipro_scores = run_ragas_eval(mipro_answers, mipro_contexts, "MIPROv2")


# FINAL RESULTS TABLE

final_results = {
    "Single-Shot": baseline_single,
    "ReAct": baseline_react,
    "BootstrapFewShot": bfs_scores,
    "MIPROv2": mipro_scores,
}
with open("final_evaluation_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print("\n" + "=" * 60)
print("FINAL ARCHITECTURE COMPARISON")
print("=" * 60)
print(f"{'Metric':<20} {'Single-Shot':<15} {'ReAct':<15} {'BFS':<15} {'MIPROv2':<15}")
print("-" * 60)
for metric in ["faithfulness", "answer_relevancy"]:
    print(
        f"{metric:<20} {final_results['Single-Shot'].get(metric, 0):<15.4f} {final_results['ReAct'].get(metric, 0):<15.4f} {final_results['BootstrapFewShot'].get(metric, 0):<15.4f} {final_results['MIPROv2'].get(metric, 0):<15.4f}"
    )
print("\nAll done! Saved to final_evaluation_results.json.")
