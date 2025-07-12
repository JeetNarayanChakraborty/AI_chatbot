# PDF-QBot

PDF-QBot is an intelligent document question-answering chatbot that allows users to upload PDF files and interact with the contents using natural language. Built using **Streamlit**, **LangChain**, **Cohere**, **FAISS**, and **PyPDF2**, this tool leverages state-of-the-art embeddings and retrieval to deliver precise answers directly from your documents.

## Features

* 📁 **PDF Upload**: Drag and drop a PDF file to initiate interaction.
* 🔍 **Text Extraction**: Automatically reads and parses the PDF contents.
* 🧹 **Chunking with Context**: Uses smart recursive character-based chunking for improved contextual understanding.
* 🧠 **Embeddings via Cohere**: Generates high-quality semantic representations of text.
* ⚡ **Vector Store with FAISS**: Efficient and scalable similarity search across document chunks.
* 💬 **Natural Language Querying**: Ask any question, and get instant answers based on the document contents.

## Installation

To set up PDF-QBot on your local machine:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/pdf-qbot.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd pdf-qbot
   ```

3. **Install the Dependencies**:

   Make sure you have Python 3.8+ installed. Then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *If you don’t have a `requirements.txt`, use:*

   ```bash
   pip install streamlit PyPDF2 langchain langchain-cohere langchain-community faiss-cpu
   ```

4. **Set Your Cohere API Key**:

   Open the script and replace the hardcoded API key with your own securely via environment variables or a `.env` file:

   ```python
   os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
   ```

5. **Run the App**:

   ```bash
   streamlit run app.py
   ```

## Usage

Once the app is running:

* **Upload a PDF** via the sidebar.
* **Type a Question** in the main panel — for example, “What is the main purpose of this document?”
* PDF-QBot will analyze and respond using the most relevant context from the uploaded file.



### 1. Semantic Search using FAISS

All PDF content is broken down into manageable chunks and indexed using **Facebook AI Similarity Search (FAISS)** for fast and scalable document retrieval.

```python
vector_store = FAISS.from_texts(chunks, embedding)
output = vector_store.similarity_search(user_question)
```

### 2. Language Model Chaining via LangChain

To ensure smooth handling of input and retrieval, **LangChain’s RetrievalQA** is used, wrapping the **Cohere LLM** into an interactive query-answering chain.

```python
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
)
response = chain.invoke({"query": user_question})
```

## Notes

* 🔐 **Security**: Do not hardcode your API keys in production environments. Use `.env` or environment variables instead.
* 📄 **PDF Size**: Optimized for small to medium PDFs. For larger documents, consider streaming or paginated chunking.



For any questions or support, please open an issue in this repository.
