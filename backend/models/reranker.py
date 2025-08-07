import os
from llama_index.core.postprocessor import SentenceTransformerRerank
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

def initialize_reranker():
    """Initializes and returns the SentenceTransformerRerank postprocessor."""
    reranker_top_n = int(os.getenv("RERANKER_TOP_N", 5)) # Default to 5 if not set

    print(f"Initializing SentenceTransformerRerank with top_n={reranker_top_n}")
    # Using a common cross-encoder for re-ranking. You can swap this model if needed.
    return SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", # A good general-purpose cross-encoder
        top_n=reranker_top_n
    )

if __name__ == "__main__":
    # Example usage:
    try:
        reranker = initialize_reranker()
        print("Reranker initialized successfully.")
    except Exception as e:
        print(f"Error initializing reranker: {e}")