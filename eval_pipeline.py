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


generator_llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.0, api_key=os.getenv("GROQ_API_KEY")
)

judge_llm = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0.0, api_key=os.getenv("GROQ_API_KEY")
)

# Gemini for embeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")
)

ragas_llm = LangchainLLMWrapper(judge_llm)
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

    import time

    vector_store = None
    batch_size = 80

    print(f"Embedding {len(final_children)} chunks in batches of {batch_size}...")

    for i in range(0, len(final_children), batch_size):
        batch = final_children[i : i + batch_size]
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, gemini_embeddings)
        else:
            vector_store.add_documents(batch)

        print(f"Embedded batch {i//batch_size + 1}... waiting 60s for rate limit.")
        time.sleep(60)

    vector_store.save_local(EVAL_INDEX_PATH)
    return vector_store, parent_lookup, parent_chunks


if os.path.exists(EVAL_INDEX_PATH):
    print(f"Loading existing Golden Dataset index...")
    vector_store = FAISS.load_local(
        EVAL_INDEX_PATH, gemini_embeddings, allow_dangerous_deserialization=True
    )
    with open(EVAL_PARENT_META, "r") as f:
        parent_lookup = json.load(f)

    # If we need to generate the dataset, we need the parent chunks
    if not os.path.exists("ragas_testset.json"):
        _, _, parent_chunks = build_eval_vector_store()
    else:
        parent_chunks = []
else:
    vector_store, parent_lookup, parent_chunks = build_eval_vector_store()

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


print("Loading manually curated 10-question dataset...")

rag_evaluation_dataset = [
    # ==========================================
    # Simple Questions (3)
    # ==========================================
    {
        "question": "What are the three deployment shapes for a document management metadata layer described in the guide, and which one serves as the reference shape?",
        "answer": "The three deployment shapes are Shape A (index over existing repositories, where content stays in the legacy repositories), Shape B (metadata layer plus object storage, where immutable content versions live in object storage buckets), and Shape C (content stored inside the database itself, such as through the GridFS convention)[cite: 11]. Shape B is the reference shape of the guide[cite: 11, 12].",
    },
    {
        "question": "Which MongoDB version syntax is used for the reference implementation in this guide, and which specific versions are required for the $rankFusion and $scoreFusion hybrid search stages?",
        "answer": "The reference implementation uses MongoDB 8.0 syntax[cite: 5]. For hybrid search, the $rankFusion stage requires MongoDB 8.0 and later, while the $scoreFusion stage requires MongoDB 8.3 and later[cite: 14, 29].",
    },
    {
        "question": "In the registry record example for the facility services contract (DOC-01J8XK4M2Q7R9TBV3N6P5A8C1E), what is the contract's monetary value, when does it expire, and what is its retention schedule ID?",
        "answer": "The contract has a value of 1,250,000 EUR, expires on 2029-09-30, and uses the retention schedule ID 'RS-CONTRACT-10Y'[cite: 20, 21].",
    },
    # ==========================================
    # Moderate Questions (3)
    # ==========================================
    {
        "question": "Why are MongoDB TTL (time-to-live) indexes unsuitable for records management disposition, and what use cases are they suitable for instead?",
        "answer": "TTL indexes are unsuitable for records management disposition because they delete documents silently after a timestamp passes without checking for legal holds, recording what was destroyed or why, deleting the underlying content object in storage, or producing an audit trail[cite: 14, 33]. Instead, TTL indexes are suitable for expiring caches, ephemeral renditions, or upload sessions[cite: 14, 33].",
    },
    {
        "question": "How does the platform's ingestion contract handle duplicate content bytes arriving under the same source key versus the exact same content bytes arriving under different source keys?",
        "answer": "In the ingestion contract, the content hash (SHA-256) identifies the bytes, whereas the combination of source system and source key identifies the item[cite: 17, 18]. When the same bytes arrive twice under the same source key, they represent a single document and duplicate intake attempts are rejected by the unique index on source.system and source.key[cite: 17, 18, 42]. When the same bytes arrive under different source keys (within the same tenant and residency zone), the platform creates two distinct registry records that point to one shared content object in the 'objects' collection and increments that object's reference count, ensuring the physical object is only deleted when its reference count reaches zero[cite: 17].",
    },
    {
        "question": "When a user's access permissions on a dossier are revoked, how does the architecture prevent unauthorized access to metadata, content, and facet counts during the asynchronous index propagation lag?",
        "answer": "During propagation lag, two real-time backstops enforce security: first, the Search API re-verifies every hit on the returned search page against the authoritative 'documents' registry collection (using majority read concern) before returning any titles, highlights, or passages[cite: 28, 36]; second, the Document API re-evaluates the full access rule at request time before issuing short-lived signed URLs for content[cite: 36, 37]. Because facet counts cannot be re-verified document by document at query time, the Search API marks facet counts as approximate while aclVersion on any affected dossier is behind, and excludes records above the 'internal' sensitivity level from facet computation altogether[cite: 28, 36, 37].",
    },
    # ==========================================
    # Complex Multi-Hop Questions (4)
    # ==========================================
    {
        "question": "In the Chapter 11 worked example, the diagnosis field of the medical report (registered as 2026/0310877) is encrypted client-side while its OCR text is projected for full-text search. Based on the capability constraints in Chapters 4 and 9, why cannot the platform enable Queryable Encryption on the 'search_docs' projection collection to support equality lookups on sensitive fields while still running full-text search, and what is the prescribed architectural alternative?",
        "answer": "According to Chapters 4 and 9, fields encrypted client-side cannot be searched via MongoDB Search, and a collection configured with Queryable Encryption cannot be indexed for MongoDB Search at all[cite: 14, 37]. Consequently, enabling Queryable Encryption on the 'search_docs' collection would prevent indexing the projected OCR text ('body') required for claims handlers to run permission-trimmed full-text and faceted searches on medical reports[cite: 26, 27, 37, 42]. To support equality or range queries on sensitive identifiers without breaking MongoDB Search, the architecture isolates those identifiers into a separate vault collection protected by Queryable Encryption (keyed by document ID), while keeping special-category fields like 'diagnosis' encrypted client-side and excluded from 'search_docs'[cite: 37, 42].",
    },
    {
        "question": "Trace how the two-phase disposition state machine in Chapter 8 (Listing 16) executes during Step 6 of the Chapter 11 worked example for the medical report (2026/0310877), explaining why Version 1's content object is physically deleted while Version 2's content object survives, and identify the accountable owner for verifying this behavior in the Chapter 13 readiness checklist.",
        "answer": "In Phase 1 of the disposition job (executed inside a multi-document transaction), the platform verifies that neither the document nor its parent dossier ('DSR-CLM-2026-18422') has active holds, transitions the registry record's lifecycle.state to 'disposing', clears current.object and current.renditions (immediately stopping the Document API from issuing new signed URLs), inventories the object keys of Version 1, Version 2, and all renditions from the 'versions' collection, decrements the reference count ('refs') for each version's hash in the 'objects' collection by 1, and inserts a tracking document into the 'dispositions' collection[cite: 33, 34, 42, 43]. In Phase 2 (executed outside the transaction after the longest signed-URL lifetime has elapsed), the job checks the 'refs' count in the 'objects' collection for each inventoried key[cite: 34, 43]. Version 1's content object reaches refs: 0 and is physically deleted from the object store[cite: 34, 43]. However, Version 2's content object is shared with a second document created when a broker uploaded the identical report via the portal under a different source key (which follows a different retention schedule); because its reference count remains greater than zero, the job leaves Version 2's bytes intact, marks the object entry as 'shared: true' in the 'dispositions' collection, deletes the renditions, and transitions the medical report to 'disposed' while preserving the metadata stub[cite: 34, 42, 43]. In the Chapter 13 readiness checklist, the accountable owner for verifying retention and disposition drills—including reference counts, URL lifetime waits, deletion proofs, and stubs—is the Records manager[cite: 48].",
    },
    {
        "question": "Compare how full-text search (Listing 10) and semantic vector search (Listing 13) structure and query data across collections, and explain the two architectural methods described in Chapter 7 to overcome the collection limitation of MongoDB's $rankFusion and $scoreFusion stages when combining them into a hybrid search.",
        "answer": "Full-text search (Listing 10) executes a $search aggregation stage against the 'search_docs' projection collection, which maintains one document per registry record containing the full extracted OCR text ('body') and denormalized metadata fields[cite: 26, 27]. Semantic vector search (Listing 13) executes a $vectorSearch stage against a separate 'chunks' collection, where each document version's text is split into passages carrying a 1024-dimension vector 'embedding' and denormalized pre-filter fields ('aclRead', 'docClass', 'state'), followed by a $group stage on '$documentId' to roll passage scores up to the document level[cite: 28, 29]. Because MongoDB's $rankFusion and $scoreFusion stages require all sub-pipelines to run against the same collection, they cannot directly combine queries across 'search_docs' and 'chunks'[cite: 14, 29]. The architecture resolves this in one of two ways: (1) the native database method adds a full-text search index ('chunks_text') directly onto the 'chunks' collection so that both the $vectorSearch and $search sub-pipelines execute within $rankFusion on 'chunks' before grouping by '$documentId' (Listing 14), or (2) the application method runs Listing 10 on 'search_docs' and Listing 13 on 'chunks' independently and merges the two ranked document lists using reciprocal rank fusion in the API layer (or via $unionWith), followed by permission re-verification against the registry[cite: 14, 29, 30].",
    },
    {
        "question": "Suppose an organization is operating in Phase 1 of the Chapter 13 adoption roadmap using Shape A over a legacy repository ('ECM-A'), and a user deletes a document in 'ECM-A' just before an older, out-of-order update event for that same document is replayed. Based on the Shape A ownership rules (Chapter 3), the ingestion contract and Listing 1 (Chapter 5), the JSON Schema validator (Chapter 6), and Phase 1 records controls (Chapter 13), what happens to the registry record, the physical content, and the replayed update event?",
        "answer": "Under Shape A, 'ECM-A' retains custody of the physical content while the metadata layer owns the lifecycle state[cite: 11]. When the document is deleted in 'ECM-A', the Chapter 5 ingestion contract dictates that a source deletion never triggers an immediate physical removal in the registry; instead, the record's lifecycle.state transitions to 'withdrawnAtSource' (an enumerated state enforced by the JSON Schema validator in Listing 7), and Phase 1 records controls explicitly disable physical destruction by default until Phase 3 records automation is implemented[cite: 17, 25, 46]. When the older, out-of-order update event from 'ECM-A' is subsequently replayed, Step 1 of Listing 1 (insertOne) fails with a 11000 duplicate key exception on the unique index { 'source.system': 1, 'source.key': 1 } and proceeds to Step 2 (updateOne)[cite: 18]. Because Step 2 enforces the positional filter { 'source.position': { $lt: evt.position } } and the replayed event's position is older than the stored position from the deletion event, the update matches zero documents, returns 'stale, ignored', and leaves the newer 'withdrawnAtSource' state untouched[cite: 17, 18, 47].",
    },
]

questions = [item["question"] for item in rag_evaluation_dataset]
references = [item["answer"] for item in rag_evaluation_dataset]


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
    ans = generator_llm.invoke(
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
    grade = generator_llm.invoke(
        f"Can you answer '{q}' using ONLY this context? Reply YES or NO.\nContext:\n{ctx}"
    ).content
    if "NO" in grade.upper():
        refined_q = generator_llm.invoke(
            f"Original question: {q}. Generate ONE new search query to find missing info."
        ).content
        new_ctx_list = retrieve_with_parents(refined_q)
        ctx_list.extend(new_ctx_list)
        ctx = "\n\n".join(ctx_list)

    ans = generator_llm.invoke(
        f"Answer using ONLY this context:\n{ctx}\n\nQuestion: {q}"
    ).content
    react_answers.append(ans)
    react_contexts.append(ctx_list)
baseline_react = run_ragas_eval(react_answers, react_contexts, "ReAct")


print("\nConfiguring DSPy (Local Phi-4)...")

dspy_lm = dspy.LM("groq/llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
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
