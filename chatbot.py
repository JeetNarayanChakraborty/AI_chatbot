import os
import streamlit as sl
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS




os.environ["COHERE_API_KEY"] = "j02SimlgDxmdPOFlLFreRs7pcTkbLodl83zI30yC"
embedding = CohereEmbeddings(model="embed-english-v3.0")

text = ""
vector_store = None
chain = None
output = None


# upload pdf files

sl.header("first chatbot")

with sl.sidebar:
    sl.title("Your documents")
    file = sl.file_uploader("Upload a pdf file and start asking questions", type="pdf")

#read each page and extract text

if file is not None:
    input_pdf = PdfReader(file)
    for page in input_pdf.pages:
        text += page.extract_text()

    # Break it into chunks

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    #Create embeddings and store embeddings for all chunk ( embed() method is called implicitly by the embedding module )
    # store chunks and their corresponding embedding in vector store

    if chunks:
        vector_store = FAISS.from_texts(chunks, embedding)

        llm = ChatCohere(
            model="command-r",
            temperature=0.7
        )


        # After embeddings are stored, we have actually trained our model, now it is ready to take questions
        # take user question

        user_question = sl.text_input("Type your question here")

        #Do similarity search

        if user_question:
            if vector_store is not None:
                output = vector_store.similarity_search(user_question)
                if output is None:
                    sl.warning("No similarity found")
            else:
                sl.warning("Please upload a PDF file first")


    # run the chain of events, take user input, take matched chunks, give user output.

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3})
            )

            response = chain.invoke({"query": user_question})
            sl.write(response["result"])


















