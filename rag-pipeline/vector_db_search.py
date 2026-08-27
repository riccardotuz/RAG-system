#!/usr/bin/env python3

# ========================================
# SECTION 4: VECTOR SEARCH
# ========================================

def search_vector_database(collection, query_embedding, top_k: int = 3):
    """
    Search vector database for relevant document chunks.
    - Vector similarity search
    - Result ranking and filtering
    - Similarity scoring
    - Top-k result selection
    """
    print("\nSECTION 4: VECTOR SEARCH")
    print("=" * 60)
    
    # Perform vector search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    print(f"> Searching for top {top_k} results")
    print(f"> Found {len(results['ids'][0])} relevant chunks") # list of matching document IDs for query #0
    
    # Process and display results
    search_results = []
    for i, (doc_id, distance, content, metadata) in enumerate(zip(
        results['ids'][0], 
        results['distances'][0], 
        results['documents'][0], 
        results['metadatas'][0]
    )):
        similarity = 1 - distance  # Convert distance to similarity (valid because Chroma's collection uses cosine space)
        search_results.append({
            'id': doc_id,
            'content': content,
            'metadata': metadata,
            'similarity': similarity
        })

        print(f"\n{i+1}. {metadata['topic']} (Source: {metadata['source']})")
        print(f"   Similarity: {similarity:.3f}")
        print(f"   Content: {content[:100]}...")
    
    return search_results