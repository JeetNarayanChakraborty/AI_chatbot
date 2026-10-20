import os
import shutil
from dotenv import load_dotenv
import streamlit as sl
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


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
        # Define how the LLM should answer the question using the context
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create the document combining chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # Combine it with the retriever
        retriever = sl.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Check if the answer is already cached
        # If it is, return the cached answer. If not, invoke the RAG chain and store the answer in the cache.
        cached_answer = semantic_cache.check(user_question)

        if cached_answer is not None:
            sl.success("⚡ Cache hit — returning a cached answer")
            sl.write(cached_answer)
            response = {"answer": cached_answer, "context": []}
        else:
            response = rag_chain.invoke({"input": user_question})
            sl.write(response["answer"])
            semantic_cache.store(user_question, response["answer"])

        with sl.expander("View Source Documents"):
            for i, doc in enumerate(response["context"]):
                sl.markdown(f"**Source {i+1}:**")
                sl.info(doc.page_content)
    else:
        sl.warning("Please upload a PDF file first")
