#!/usr/bin/env python3

# ========================================
# SECTION 6: RESPONSE GENERATION
# ========================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_response(llm, augmented_prompt: str) -> str:
    """
    Generate response using LLM via OpenRouter.
    """
    print("\nSECTION 6: RESPONSE GENERATION")
    print("=" * 60)
    print("> Processing with LLM...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful IT helpdesk assistant."),
        ("user", "{augmented_prompt}")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"augmented_prompt": augmented_prompt})
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        response = "Sorry, I couldn't generate a response right now. Please try again."
    
    print(f"> Generated response length: {len(response)} characters")
    
    return response