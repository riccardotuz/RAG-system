#!/usr/bin/env python3

# ========================================
# SECTION 1: DOCUMENT LOADING & CHUNKING
# ========================================

"""
Agentic Chunking is an advanced chunking method. Instead of splitting by character count or sentence boundaries, an AI model analyzes the document and decides optimal split points based on topic shifts and semantic coherence.
Agentic chunking uses an LLM to understand the meaning and structure of documents, creating chunks that preserve complete topics and ideas.
Using LLM to intelligently split documents based on semantic meaning.

Benefits:
- AI analyzing document for topic boundaries (semantic splitting)
- Semantically coherent chunks (one topic per chunk)

Note: API access is required.
"""

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Agentic chunking using LLM
def agentic_chunking(llm, text):
    """
    Uses an LLM to split text into semantically distinct chunks.
    The AI analyzes semantic topic shifts and creates meaningful boundaries.
    """

    # The Prompt: Instruct the LLM to act as a "Chunking Agent"
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert document editor specializing in semantic document analysis.
        Your task is to split the provided text into semantically distinct chunks based on topic shifts.

        Rules:
        1. Keep related sentences together - don't break up a single topic
        2. Split ONLY when the topic changes significantly
        3. Each chunk should be about ONE coherent topic
        4. Output the chunks separated by '---SPLIT---'
        5. Do not modify the original text - just split it at appropriate boundaries
        6. Include section headers with their content in the same chunk"""),
        ("user", "{text}")
    ])

    # Chain the prompt, LLM and output parser to return a plain-text response (LangChain's pipeline syntax: the output of each component becomes the input to the next)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"text": text})
        # Split the response by our delimiter and clean up
        chunks = [c.strip() for c in response.split("---SPLIT---") if c.strip()]
        return chunks
    except Exception as e:
        print(f"\n❌ API Error: {e}")
        return []

'''
The csv file is not one long document, it is 10 separate knowledge-item documents (one per row), each already about a single topic.
So, instead of calling agentic_chunking() once on a sample document, we loop over the rows and call it once per row.
'''

def load_and_chunk_documents(llm):

    print("SECTION 1: DOCUMENT LOADING & CHUNKING")
    print("=" * 60)

    # ========================================
    # Document Loading: load the test dataset from Kaggle
    # "dkhundley/sample-rag-knowledge-item-dataset" - https://www.kaggle.com/datasets/dkhundley/sample-rag-knowledge-item-dataset
    # ========================================
    
    # Columns: ki_topic, ki_text, sample_question, sample_ground_truth
    # Each ROW is its own self-contained knowledge-item document
    csv_path = "../rag_sample_qas_from_kis.csv"

    df = pd.read_csv(csv_path)

    print(f"> Loaded {len(df)} knowledge-item documents from CSV")
    print()

    # ========================================
    # Document Chunking: run agentic chunking over every document in the CSV
    # ========================================

    all_chunks = [] # list of dicts

    # Loop over the CSV rows
    for idx, row in df.iterrows():
        topic = row["ki_topic"]
        document_text = row["ki_text"]

        # Call agentic_chunking() on each row
        doc_agentic_chunks = agentic_chunking(llm, document_text)

        print(f"[{idx + 1}/{len(df)}] Chunking: {topic}")
        print(f"Length: {len(document_text)} characters")

        if not doc_agentic_chunks:
            print(f"No chunks produced, skipping.\n")
            continue

        print(f"> Produced {len(doc_agentic_chunks)} chunk(s)")

        # Each resulting chunk is stored as a dict
        for chunk_idx, chunk in enumerate(doc_agentic_chunks, 1):
            preview = chunk[:80].replace('\n', ' ').strip()
            print(f"   Preview: {preview}...")
            print()

            all_chunks.append({
                "source_topic": topic,
                "row_index": idx,
                "chunck_index": chunk_idx,
                "content": chunk,
                # Carried along for a later retrieval-evaluation step, not used in chunking itself
                "sample_question": row["sample_question"],
                "sample_ground_truth": row["sample_ground_truth"]
            })
        
    print("\n> Agentic chunking completed successfully!")
    print(f"> Created {len(all_chunks)} total chunks (semantic-based) from {len(df)} documents")

    return all_chunks