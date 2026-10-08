import os
from dotenv import load_dotenv
import streamlit as sl
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


cohere_api_key = os.getenv("COHERE_API_KEY")
os.environ["COHERE_API_KEY"] = cohere_api_key

embedding = CohereEmbeddings(model="embed-english-v3.0")
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

    # Store chunks and their corresponding embedding in vector store
    if chunks:
        sl.session_state.vector_store = FAISS.from_texts(chunks, embedding)
        sl.session_state.processed_file = file.name


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

        # Invoke the chain using the new keys ('input' and 'answer')
        # The retriever inside this chain automatically does the similarity search
        response = rag_chain.invoke({"input": user_question})
        sl.write(response["answer"])
    else:
        sl.warning("Please upload a PDF file first")
