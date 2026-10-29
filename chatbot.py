import os
import re
import shutil
import json
from dotenv import load_dotenv
import streamlit as sl
from PyPDF2 import PdfReader
import pymupdf4llm
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

FAISS_INDEX_PATH = "faiss_index"
PROCESSED_FILE_META = "processed_file.txt"


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


gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = gemini_api_key

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
semantic_cache = SemanticCache(embedding=embedding, similarity_threshold=0.90)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY"),
)


# Node A: Retrieve (with Parent-Child swapping)
def retrieve_node(state: AgentState):
    query = state.get("refined_query") or state["question"]

    # 1. Search the tiny Child chunks (k=4 for better coverage)
    new_docs = sl.session_state.vector_store.as_retriever(
        search_kwargs={"k": 4}
    ).invoke(query)

    # 2. Load the Parent lookup dictionary
    parent_lookup = {}
    if os.path.exists("parent_lookup.json"):
        with open("parent_lookup.json", "r") as f:
            parent_lookup = json.load(f)

    # 3. Swap Children for Parents
    parent_docs = []
    seen_parents = set()

    for doc in new_docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id and parent_id not in seen_parents:
            # Grab the big parent text
            parent_content = parent_lookup.get(parent_id, doc.page_content)
            # Create a new document with the parent content
            parent_doc = Document(page_content=parent_content, metadata=doc.metadata)
            parent_docs.append(parent_doc)
            seen_parents.add(parent_id)
        elif not parent_id:
            parent_docs.append(doc)  # Fallback if no parent_id

    # 4. Accumulate: keep old docs, add only unique new ones
    existing = state.get("documents", [])
    seen = {d.page_content for d in existing}
    added = 0
    for d in parent_docs:
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
    context = "\n\n".join(
        f"[Source {i+1}]\n{d.page_content}" for i, d in enumerate(docs)
    )
    coverage = state["coverage_history"][-1] if state["coverage_history"] else 1.0

    system_prompt = (
        "You answer questions using ONLY the provided context.\n\n"
        "Rules:\n"
        "1. Answer directly and concisely. No preamble, no filler.\n"
        "2. You MUST cite sources inline using [Source N] next to each claim. "
        "For example: 'Jeet works at Deloitte [Source 1].'\n"
        "3. If the context covers only part of a multi-part question, "
        "answer the parts you can and explicitly state what the document does not cover.\n"
        "4. If the context does not contain the answer, reply exactly: "
        '"I don\'t know based on the provided document."\n'
        "5. Never use outside knowledge. Never invent facts, numbers, or source tags.\n"
    )
    if coverage < 0.8:
        system_prompt += (
            f"\nNote: retrieval estimated only {coverage:.0%} coverage. "
            "Parts of the question may not be answerable from the context. "
            "Be explicit about what is missing.\n"
        )
    system_prompt += f"\nContext:\n{context}"

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


# Restore vector store
if sl.session_state.vector_store is None and os.path.exists(FAISS_INDEX_PATH):
    try:
        sl.session_state.vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding,
            allow_dangerous_deserialization=True,
        )
        # Restore the filename
        if os.path.exists(PROCESSED_FILE_META):
            with open(PROCESSED_FILE_META, "r") as f:
                sl.session_state.processed_file = f.read().strip()
    except Exception:
        # Corrupted or incompatible index — start fresh
        sl.session_state.vector_store = None
        sl.session_state.processed_file = None


# Upload pdf files
sl.header("first chatbot")

with sl.sidebar:
    sl.title("Your documents")
    file = sl.file_uploader("Upload a pdf file and start asking questions", type="pdf")

# Read each page and extract text
if file is not None and sl.session_state.processed_file != file.name:

    #  Save the uploaded file temporarily so pymupdf can read it
    temp_path = f"temp_{file.name}"
    with open(temp_path, "wb") as f:
        f.write(file.getbuffer())

    with sl.spinner("Parsing PDF structure..."):
        #  Convert PDF to Markdown (Structure-Aware)
        md_text = pymupdf4llm.to_markdown(temp_path)
        os.remove(temp_path)  # Clean up temp file

    #  Split by Headers to create PARENTS
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    parent_chunks = markdown_splitter.split_text(md_text)

    #  Create CHILDREN and the Parent Lookup
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

    final_children = []
    parent_lookup = {}

    for i, parent in enumerate(parent_chunks):
        parent_id = f"parent_{i}"

        # Save the big text for later
        parent_lookup[parent_id] = parent.page_content

        # Chop this parent into tiny children
        children = child_splitter.split_documents([parent])
        for child in children:
            # Attach the sticky note pointing to the parent
            child.metadata["parent_id"] = parent_id
            final_children.append(child)

    # Save the lookup dictionary to disk
    with open("parent_lookup.json", "w") as f:
        json.dump(parent_lookup, f)

    #  Store chunks and their corresponding embedding in vector store (batched)
    if final_children:
        import time

        vector_store = None
        batch_size = 80

        # Loop through the CHILDREN in batches
        for i in range(0, len(final_children), batch_size):
            batch = final_children[i : i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embedding)
            else:
                vector_store.add_documents(batch)

            time.sleep(60)

        sl.session_state.vector_store = vector_store
        sl.session_state.processed_file = file.name

        # Persist to disk
        vector_store.save_local(FAISS_INDEX_PATH)
        with open(PROCESSED_FILE_META, "w") as f:
            f.write(file.name)

        semantic_cache.clear()


# Take user question
user_question = sl.text_input("Type your question here")


# Do similarity search
if user_question:
    if sl.session_state.vector_store is not None:

        # 1. Check semantic cache first
        cached_answer = semantic_cache.check(user_question)

        if cached_answer is not None:
            sl.success("Cache hit — returning cached answer")
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

            # Find all citations in the answer
            cited_sources = re.findall(r"\[Source (\d+)\]", final_state["answer"])

            # Convert to integers, remove duplicates, and sort them
            cited_indices = sorted(list(set(int(i) - 1 for i in cited_sources)))

            with sl.expander("View Source Documents"):
                # If the LLM cited specific sources, only show those.
                # If it didn't cite any, show all retrieved docs as a fallback.
                if cited_indices:
                    for idx in cited_indices:
                        if idx < len(final_state["documents"]):
                            doc = final_state["documents"][idx]
                            sl.markdown(f"**Source {idx + 1}:**")
                            sl.info(doc.page_content)
                else:
                    for i, doc in enumerate(final_state["documents"]):
                        sl.markdown(f"**Source {i+1}:**")
                        sl.info(doc.page_content)

            # 4. Cache the final answer
            semantic_cache.store(user_question, final_state["answer"])
    else:
        sl.warning("Please upload a PDF file first")
