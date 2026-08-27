#!/usr/bin/env python3

# ========================================
# SECTION 2: VECTOR DATABASE SETUP
# ========================================

from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions

def get_vector_collection(db_path: str = "./chroma_db", collection_name: str = "IT_helpdesk_articles"):
    
    # Initialize ChromaDB client (our connection to the vector database)
    client = chromadb.PersistentClient(path=db_path) # Persist the database to disk (save vector database to file)
    
    # Embedding function (all-MiniLM-L6-v2 is Chroma's default embedding model which converts text to 384-dimensional vectors)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # Create a collection (a container for our documents, like a table in SQL) or get it if collection already exists
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"} # similarity metric
    )

    return collection

def setup_vector_database(chunks: List[Dict]):
    """
    Set up ChromaDB vector database and store document chunks.
    """
    print("\nSECTION 2: VECTOR DATABASE SETUP")
    print("=" * 60)
    
    collection = get_vector_collection()
    
    print(f"> Created/Loaded collection: {collection.name}")
    print(f"> Similarity metric: cosine")

    # Document embedding and storage
    
    # Prepare data for storage
    # unique index as a string computed from: row_index (document) + chunk_index (chunk within that document)
    row_chunk_indexes = [str(chunk["row_index"])+str(chunk["chunck_index"]) for chunk in chunks] # cast to str for collection.add()
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [{"topic": chunk["source_topic"], "source": chunk["row_index"]} for chunk in chunks]
    
    # Add documents to collection (embeddings will be generated automatically from chunks)
    if collection.count() == 0:
        collection.add(
            ids=row_chunk_indexes,
            documents=documents,
            metadatas=metadatas
        )
        print(f"> Stored {len(chunks)} chunks in vector database")
    else:
        print(f"> Collection already contains {collection.count()} chunks")
    
    print(f"> Collection count: {collection.count()}")

    print("> ChromaDB Vector Database Initialized Successfully!")
    
    return collection