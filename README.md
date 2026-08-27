# RAG (Retrieval-Augmented Generation) system

## Features
* Document chunking (Agentic Chunking)
* Vector database storage + embedding with ChromaDB ("all-MiniLM-L6-v2" embedding model)
* User query processing
* Vector search
* Context augmentation + Response generation

### 1. Setup Environment

A virtual environment isolates your project's packages from other Python projects, avoiding version conflicts:

```bash
# Clone the repository
git clone https://github.com/riccardotuz/RAG-system.git
# Navigate to the project
cd RAG-system/rag-pipeline
# Create a virtual enviornment
python3 -m venv venv
# Activate the virtual environment. You will see (venv) in your terminal when active.
source venv/bin/activate
# Now install the required Python packages (see 2. Installation)
```

> [!IMPORTANT]
> The RAG system implementation connects to a language model via [OpenRouter](https://openrouter.ai/).
> 
> **What is OpenRouter?** It is a service that provides unified access to multiple LLM providers (OpenAI, Anthropic, etc.) through a single API, making it easy to switch between models.
> 
> Thus, you need to obtain an OpenRouter API key first. To keep your API key secure, it will be stored in a `.env` file (that will not be committed to version control).
> 
> Create a `.env` file in the project root directory (/RAG-system):
> 
> `OPENROUTER_API_KEY=your-api-key-here`
> 
> Replace `your-api-key-here` with the key you copied from OpenRouter.

### 2. Installation

Within the virtual environment, install the required Python packages directly. This cell is self-contained, no external `requirements.txt` needed:

```bash
# Activate the virtual environment (if not already active)
source venv/bin/activate
# Install dependencies
pip install python-dotenv langchain-openai langchain-core pandas chromadb sentence-transformers
```

* `python-dotenv` - For loading environment variables from the `.env` file
* `langchain-openai` - Integration for OpenAI-compatible APIs (including OpenRouter)
* `langchain-core` - For Agentic Chunking
* `pandas` - For document loading/reading
* `chromadb` - For vector database
* `sentence-transformers` - For embedding models

---

### RAG pipeline explained

#### 1) Document Loading & Chunking 
Documents are loaded and chunked to improve retrieval precision.

It implements **Agentic Chunking** which uses an LLM to understand the meaning and structure of documents, creating chunks that preserve complete topics and ideas. So, instead of splitting by character count or sentence boundaries, an LLM model analyzes the document and decides optimal split points based on semantic meaning and topic shifts.
```
document_chunking.py
```
#### 2) Vector Databse Setup + Embedding
The vector database is set up to store the chunked documents (converted to embeddings). It shows ChromaDB initialization and collection creation with specific configuration. It uses the cosine similarity metric in the ChromaDB collection.
```
vector_db_setup.py
```
#### 3) Query Processing
User queries are processed and converted to embeddings for vector search. The `all-MiniLM-L6-v2` embedding model (loaded locally) is used for query processing. This embedding model produces 384 dimensions. The Query Processing component is responsible for converting user questions into searchable vectors.
```
user_query_processing.py
```
#### 4) Vector Database Search
Vector search finds relevant document chunks using semantic similarity.
```
vector_db_search.py
```
#### 5) Context Augmentation
Retrieved context is assembled, from search results, into a structured prompt for the LLM (augment prompt with context). It shows how multiple sources are combined into coherent context.
```
context_augmentation.py
```
#### 6) Response Generation (Generate response using LLM)
The LLM generates responses using the augmented prompt. This is the final step of the RAG pipeline.
```
response_generation.py
```
#### 7) Complete RAG Pipeline
This is the complete RAG pipeline in action. Here is where all the components work together, from document loading to response generation.
```
main.py
```

---

### Running the pipeline

To run the complete RAG pipeline, within the virtual environment, run `main.py`:
```
python3 main.py
```