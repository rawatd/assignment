# C:\Assignment_2\backend\rag_pipeline.py
import os
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

from backend.models.llm_client import initialize_ollama_llm
from backend.models.reranker import initialize_reranker
from backend.services.db_connection import get_vector_store, setup_pgvector_extension # Ensure this import is correct

load_dotenv() # Load environment variables from .env

# Global variables to hold initialized components
rag_query_engine = None
embed_model_instance = None
llm_instance = None
reranker_instance = None

def initialize_rag_pipeline():
    """Initializes all components of the RAG pipeline."""
    global rag_query_engine, embed_model_instance, llm_instance, reranker_instance

    if rag_query_engine is not None:
        print("RAG pipeline already initialized.")
        return rag_query_engine

    print("Initializing RAG pipeline components...")

    # 1. Ensure PGVector extension is set up
    setup_pgvector_extension()

    # 2. Initialize LLM
    llm_instance = initialize_ollama_llm()
    Settings.llm = llm_instance # Set global LLM for LlamaIndex

    # 3. Initialize Embedding Model
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    if not embedding_model_name:
        raise ValueError("EMBEDDING_MODEL_NAME must be set in .env")
    embed_model_instance = HuggingFaceEmbedding(model_name=embedding_model_name)
    Settings.embed_model = embed_model_instance # Set global Embedding Model for LlamaIndex

    # 4. Initialize PGVector store
    # This now correctly uses PG_TABLE_NAME and EMBEDDING_DIM from .env via get_vector_store()
    vector_store = get_vector_store() 

    # 5. Load LlamaIndex from existing vector store
    print("Loading LlamaIndex from existing vector store...")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    print("LlamaIndex loaded successfully.")

    # 6. Initialize Re-ranker
    reranker_instance = initialize_reranker() # This uses RERANKER_TOP_N from .env

    # 7. Configure LlamaIndex Query Engine with re-ranking
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", 10))
    
    rag_query_engine = index.as_query_engine(
        similarity_top_k=retrieval_top_k, # Retrieve more initial chunks
        node_postprocessors=[reranker_instance], # Then re-rank to top_n
        llm=llm_instance # Explicitly pass LLM for generation
    )
    print("RAG query engine configured.")
    return rag_query_engine

def get_rag_query_engine():
    """Returns the initialized RAG query engine."""
    if rag_query_engine is None:
        initialize_rag_pipeline()
    return rag_query_engine

def get_llm_instance():
    """Returns the initialized LLM instance."""
    if llm_instance is None:
        initialize_rag_pipeline()
    return llm_instance

if __name__ == "__main__":
    # Example usage:
    try:
        query_engine = initialize_rag_pipeline()
        print("\nRAG Pipeline initialized. You can now use query_engine to ask questions.")
        # Example query (ensure your DB has data from Step 4)
        # response = query_engine.query("What are the key security policies for information access?")
        # print(response)
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")