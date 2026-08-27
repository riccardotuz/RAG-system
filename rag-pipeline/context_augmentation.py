#!/usr/bin/env python3

# ========================================
# SECTION 5: CONTEXT AUGMENTATION
# ========================================

from typing import List, Dict

def augment_prompt_with_context(query: str, search_results: List[Dict]) -> str:
    """
    Build augmented prompt with retrieved context for LLM.
    - Context assembly from search results
    - Prompt construction
    - Information formatting
    - Context length management
    """
    print("\nSECTION 5: CONTEXT AUGMENTATION")
    print("=" * 60)
    
    # Assemble context from search results
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(f"Source {i}: {result['metadata']['topic']}\n{result['content']}")
    
    context = "\n\n".join(context_parts)
    
    print(f"> Assembled context from {len(search_results)} sources")
    print(f"> Context length: {len(context)} characters")
    
    # Build augmented prompt (notice the grounding in the last part of the prompt)
    augmented_prompt = f"""
Based on the following IT helpdesk articles, answer the user's question.

IT HELPDESK ARTICLES:
{context}

QUESTION: {query}

Please provide a clear, accurate answer based on the information provided in the articles above.
If the information is not available in the articles, say so instead of guessing.
Include relevant article details and any limitations or requirements.
"""
    
    print(f"> Augmented prompt length: {len(augmented_prompt)} characters")
    print(f"> Context sources: {[result['metadata']['topic'] for result in search_results]}")
    
    return augmented_prompt