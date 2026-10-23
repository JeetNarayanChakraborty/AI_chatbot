import os
import shutil
import json
from dotenv import load_dotenv
import streamlit as sl
from PyPDF2 import PdfReader
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    refined_query: str
    iterations: int
    coverage_history: List[float]  # coverage score after each grading
    previous_queries: List[str]  # all queries issued so far
    tokens_used: int  # rough token accounting
    answer: str  # final answer from generate_node


class SemanticCache:
    def __init__(
        self, embedding, similarity_threshold=0.90, cache_path="semantic_cache"
    ):
        self.embedding = embedding
        self.threshold = similarity_threshold
        self.cache_path = cache_path
        self.index = None

        # Try to load a previously saved cache from disk
        try:
            self.index = FAISS.load_local(
                cache_path, embedding, allow_dangerous_deserialization=True
            )
        except Exception:
            self.index = None

    def check(self, query: str):
        if self.index is None:
            return None

        results = self.index.similarity_search_with_score(query, k=1)

        if not results:
            return None

        doc, l2_distance = results[0]

        cos_sim = 1 - (l2_distance**2) / 2  # cosine_similarity = 1 - (L2_distance²) / 2
        if cos_sim >= self.threshold:
            return doc.metadata.get("answer")

        return None

    def store(self, query: str, answer: str):
        doc = Document(page_content=query, metadata={"answer": answer})

        if self.index is None:
            self.index = FAISS.from_documents([doc], self.embedding)
        else:
            self.index.add_documents([doc])

        self.index.save_local(self.cache_path)

    def clear(self):
        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path)
        self.index = None


load_dotenv()


cohere_api_key = os.getenv("COHERE_API_KEY")
os.environ["COHERE_API_KEY"] = cohere_api_key

embedding = CohereEmbeddings(model="embed-english-v3.0")
semantic_cache = SemanticCache(embedding=embedding, similarity_threshold=0.90)
llm = ChatCohere(model="command-r7b-12-2024", temperature=0.7)


# Node A: Retrieve (with accumulation + exhaustion detection)
def retrieve_node(state: AgentState):
    query = state.get("refined_query") or state["question"]

    new_docs = sl.session_state.vector_store.as_retriever(
        search_kwargs={"k": 3}
    ).invoke(query)

    # Accumulate: keep old docs, add only unique new ones
    existing = state.get("documents", [])
    seen = {d.page_content for d in existing}
    added = 0
    for d in new_docs:
        if d.page_content not in seen:
            existing.append(d)
            seen.add(d.page_content)
            added += 1

    # If nothing new was added, mark exhausted for Signal 2
    new_refined = state.get("refined_query") if added > 0 else "__EXHAUSTED__"

    return {
        "documents": existing,
        "refined_query": new_refined,
        "previous_queries": state["previous_queries"] + [query],
        "tokens_used": state["tokens_used"] + (len(query) + 1500) // 4,
    }


# Node B: Grade (structured coverage scoring)
def grade_node(state: AgentState):
    question = state["question"]
    docs = state["documents"]
    context = "\n\n".join(d.page_content for d in docs)

    grader_system = (
        "You are a strict coverage grader. Given a user question and retrieved context, "
        "estimate what fraction (0.0 to 1.0) of the question can be answered from the context. "
        "If coverage < 1.0, provide ONE specific search query that would find the missing piece. "
        "If coverage is already 1.0, set refined_query to null.\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{{"coverage": <float between 0.0 and 1.0>, "refined_query": "<query string or null>"}}'
    )

    messages = [
        ("system", grader_system),
        ("human", f"Question: {question}\n\nContext:\n{context}"),
    ]

    try:
        raw = llm.invoke(messages).content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        coverage = float(parsed.get("coverage", 0.0))
        refined = parsed.get("refined_query")
    except Exception:
        # If parsing fails, exit to generate with what we have
        coverage = 1.0
        refined = None

    return {
        "refined_query": refined,
        "coverage_history": state["coverage_history"] + [coverage],
        "previous_queries": state["previous_queries"] + ([refined] if refined else []),
        "iterations": state["iterations"] + 1,
    }


# Node C: Generate
def generate_node(state: AgentState):
    question = state["question"]
    docs = state["documents"]
    context = "\n\n".join(d.page_content for d in docs)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        f"Context:\n{context}"
    )
    messages = [
        ("system", system_prompt),
        ("human", question),
    ]
    answer = llm.invoke(messages).content
    return {"answer": answer}


# Routing logic — four termination signals
def should_continue(state: AgentState):
    coverage = state["coverage_history"]
    latest = coverage[-1] if coverage else 0.0

    # Signal 1: successful completion — coverage >= 95%
    if latest >= 0.95:
        return "generate"

    # Signal 2: retrieval exhaustion — no new chunks found
    if state.get("refined_query") == "__EXHAUSTED__":
        return "generate"

    # Signal 3a: coverage stagnation — < 2% improvement
    if len(coverage) >= 2 and latest <= coverage[-2] + 0.02:
        return "generate"

    # Signal 3b: query stagnation — refined query duplicates an earlier one
    if (
        state.get("refined_query")
        and state["refined_query"] in state["previous_queries"][:-1]
    ):
        return "generate"

    # Signal 4: token budget backstop (~80k tokens)
    if state["tokens_used"] >= 80_000:
        return "generate"

    return "retrieve"


# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    should_continue,
    {"retrieve": "retrieve", "generate": "generate"},
)
workflow.add_edge("generate", END)

agentic_rag = workflow.compile()

# Initialize session state variables
if "vector_store" not in sl.session_state:
    sl.session_state.vector_store = None

if "processed_file" not in sl.session_state:
    sl.session_state.processed_file = None


# Upload pdf files
sl.header("first chatbot")

with sl.sidebar:
    sl.title("Your documents")
    file = sl.file_uploader("Upload a pdf file and start asking questions", type="pdf")

# Read each page and extract text
if file is not None and sl.session_state.processed_file != file.name:
    text = ""
    input_pdf = PdfReader(file)
    for page in input_pdf.pages:
        text += page.extract_text()

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=1000, chunk_overlap=150, length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Store chunks and their corresponding embedding in vector store (batched)
    if chunks:
        import time

        # Create an empty vector store first
        vector_store = None
        batch_size = 320  # Process 320 chunks (≈80k tokens) at a time

        # Loop through the chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embedding)
            else:
                vector_store.add_texts(batch)

            # Wait 60 seconds between batches to respect the per-minute rate limit
            time.sleep(60)

        sl.session_state.vector_store = vector_store
        sl.session_state.processed_file = file.name
        semantic_cache.clear()


# Take user question
user_question = sl.text_input("Type your question here")


# Do similarity search
if user_question:
    if sl.session_state.vector_store is not None:

        # 1. Check semantic cache first
        cached_answer = semantic_cache.check(user_question)

        if cached_answer is not None:
            sl.success("⚡ Cache hit — returning cached answer")
            sl.write(cached_answer)
        else:
            # 2. Cache miss → run the agentic ReAct loop
            initial_state = {
                "question": user_question,
                "documents": [],
                "refined_query": None,
                "iterations": 0,
                "coverage_history": [],
                "previous_queries": [],
                "tokens_used": 0,
                "answer": "",
            }

            with sl.spinner("Thinking and retrieving..."):
                final_state = agentic_rag.invoke(initial_state)

            # 3. Show the answer + how the loop arrived at it
            sl.write(final_state["answer"])

            with sl.expander(
                f" Agent reasoning (looped {final_state['iterations']} times)"
            ):
                sl.write(
                    f"**Coverage per iteration:** {final_state['coverage_history']}"
                )
                sl.write(f"**Queries issued:** {final_state['previous_queries']}")

            with sl.expander("📚 View Source Documents"):
                for i, doc in enumerate(final_state["documents"]):
                    sl.markdown(f"**Source {i+1}:**")
                    sl.info(doc.page_content)

            # 4. Cache the final answer
            semantic_cache.store(user_question, final_state["answer"])
    else:
        sl.warning("Please upload a PDF file first")
