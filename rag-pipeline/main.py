#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from document_loading_and_chunking import load_and_chunk_documents
from vector_db_setup import setup_vector_database, get_vector_collection
from user_query_processing import process_user_query
from vector_db_search import search_vector_database
from context_augmentation import augment_prompt_with_context
from response_generation import generate_response

# Load environment variables from .env file
load_dotenv()
# API key is loaded from .env file
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    print("Error: OPENROUTER_API_KEY not found.")
    print("Please ensure the environment is configured correctly.")
    sys.exit(1)

# Configure OpenRouter model
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0, # Deterministic output for consistency
    max_tokens=1000,
)

# ========================================
# COMPLETE RAG PIPELINE
# ========================================

def run_complete_rag_pipeline(query: str, llm ,collection):
    """
    Run the complete RAG pipeline from start to finish:
    
    1. Document loading and chunking
    2. Vector database setup
    3. Query processing
    4. Vector search
    5. Context augmentation
    6. Response generation
    """

    # Step 1 & Step 2 performed outside (one-time setup)

    # Step 3: Process user query
    query_embedding = process_user_query(query)

    # Step 4: Search vector database
    search_results = search_vector_database(collection, query_embedding)

    # Step 5: Augment prompt with context
    augmented_prompt = augment_prompt_with_context(query, search_results)

    # Step 6: Generate response
    response = generate_response(llm, augmented_prompt)

    # Display final result
    print("\nFINAL RESULT")
    print("=" * 60)
    print(response)
    print("=" * 60)
    
    return response

# ========================================
# MAIN EXECUTION
# ========================================

print("IT Helpdesk Assistant - Complete RAG Pipeline Demo")
print("=" * 60)

# One-time setup: Check if collection exists and contains document chunks
collection = get_vector_collection()

if collection.count() > 0:
    print(f"\n> Found existing ChromaDB collection '{collection.name}' with {collection.count()} chunks.")
    print("> Skipping document loading, agentic chunking and vector DB setup.")
else:
    print("\n> Collection is empty. Executing document loading and agentic chunking...")
    # Step 1: Load and chunk documents
    chunks = load_and_chunk_documents(llm)
    # Step 2: Setup vector database
    collection = setup_vector_database(chunks)

# Test queries (from Kaggle dataset)
test_queries = [
    "How do I set up my company email on my mobile device?",
    "How do I set up VPN access on my laptop so I can work from home and access company resources?",
    "What steps can I take to fix my printer when it's jammed and won't print?",
    "How do I set up a secure wireless network for my workgroup to prevent unauthorized access and ensure data protection?"
]

print("\nCOMPLETE RAG PIPELINE DEMO")

# Run demo for each query
for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*60}")
    print(f"DEMO {i}")
    print(f"> User Question: {query}")
    print(f"{'='*60}")
    
    try:
        run_complete_rag_pipeline(query, llm, collection)
    except Exception as e:
        print(f"Error in demo {i}: {e}")
    
    if i < len(test_queries):
        input("\n> Press Enter to continue to next demo...")

print("\nAll demos completed!")