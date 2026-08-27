#!/usr/bin/env python3

# ========================================
# SECTION 3: QUERY PROCESSING
# ========================================

from sentence_transformers import SentenceTransformer

def process_user_query(query: str):
    """
    Process user query and convert to embedding for vector search.
    """
    print("\nSECTION 3: USER QUERY PROCESSING")
    print("=" * 60)
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"> Using model: {model}")
    #print(f"> Embedding dimensions: {model.get_embedding_dimension()}")
    
    # Preprocess query
    cleaned_query = query.lower().strip()
    #print(f"Original query: '{query}'")
    #print(f"Cleaned query: '{cleaned_query}'")
    
    # Convert query to embedding
    query_embedding = model.encode([cleaned_query])
    #print(f"Query embedding shape: {query_embedding.shape}")
    #print(f"Embedding sample: {query_embedding[0][:5]}...")
    
    return query_embedding[0]