import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    answer_similarity,
    context_recall,
    context_precision,
)
import time
from functools import wraps

# LlamaIndex imports for PGvector and Ollama integration
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# RAGas specific wrappers for LlamaIndex LLM/Embedding compatibility
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import Langchain's Ollama for RAGas evaluation
from langchain_community.llms import Ollama as LangchainOllama
from langchain_community.embeddings import OllamaEmbeddings as LangchainOllamaEmbeddings


# Import nest_asyncio to handle potential event loop issues in certain environments
import nest_asyncio
nest_asyncio.apply()

# --- Configuration for your PGvector Database (from .env) ---
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")
DB_NAME = os.getenv("PG_DATABASE", "uae_db")
DB_USER = os.getenv("PG_USER", "dev")
DB_PASSWORD = os.getenv("PG_PASSWORD", "dev")
DB_TABLE_NAME = os.getenv("PG_TABLE_NAME", "my_uae_hr_laws_embeddings")

# --- Ollama LLM and Embedding Configuration (from .env) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma2:2b")
EMBEDDING_MODEL = "all-minilm:33m" 
EMBEDDING_DIM = 384

# --- Configure LlamaIndex global settings ---
Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=3600.0)
Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=3600.0)

# A simple decorator to handle retries with exponential backoff for a function
def retry_with_exponential_backoff(max_retries=3, base_delay=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except TimeoutError as e:
                    delay = base_delay * (2 ** i)
                    print(f"[{func.__name__}] TimeoutError. Retrying in {delay} seconds... (Attempt {i+1}/{max_retries})")
                    time.sleep(delay)
            print(f"[{func.__name__}] Max retries ({max_retries}) exceeded.")
            raise
        return wrapper
    return decorator

@retry_with_exponential_backoff(max_retries=3, base_delay=5)
def perform_query(query_engine, question):
    """A wrapper function to query the RAG pipeline with a retry mechanism."""
    return query_engine.query(question)

def initialize_rag_pipeline():
    """
    Initializes the LlamaIndex RAG pipeline with PGvector as the vector store.
    """
    print(f"Connecting to PGvector: host={DB_HOST}, port={DB_PORT}, dbname={DB_NAME}, user={DB_USER}, table_name={DB_TABLE_NAME}")
    try:
        vector_store = PGVectorStore.from_params(
            database=DB_NAME,
            host=DB_HOST,
            password=DB_PASSWORD,
            port=int(DB_PORT),
            user=DB_USER,
            table_name=DB_TABLE_NAME,
            embed_dim=EMBEDDING_DIM,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        query_engine = index.as_query_engine()
        print("LlamaIndex RAG pipeline initialized successfully with Ollama and PGvector.")
        return query_engine
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        print("Please ensure your PGvector database is running and accessible with the correct credentials,")
        print(f"and that the table '{DB_TABLE_NAME}' exists and is populated.")
        return None

def evaluate_rag_system_with_pipeline():
    """
    Evaluates the LlamaIndex RAG system using RAGas metrics.
    It dynamically queries the RAG pipeline to get answers and contexts.
    """

    # Initialize your RAG pipeline
    query_engine = initialize_rag_pipeline()
    if query_engine is None:
        print("Cannot proceed with evaluation as RAG pipeline failed to initialize.")
        return

    # Define your evaluation dataset with questions and ground truths as a list of dictionaries
    # THIS IS THE NEW DATA YOU PROVIDED
    eval_data_points = [
        {
            'question': "Definition of the Procurement",
            'ground_truths': ["The Procurement is the support function that ensures identification, sourcing, and access to optimal goods, services and projects that the government entities and its End-users require to fulfil their objectives.The Procurement exists to explore supply market opportunities and to ensure optimal implementation of the sourcing strategies that deliver the best possible supply outcomes for the government entity."]
        }
    ]
    
    # Initialize dictionaries for DataFrame columns
    eval_data = {
        'question': [],
        'ground_truths': [],
        'answer': [],
        'contexts': [],
        'reference': []
    }
    
    print("\nRunning queries through your RAG pipeline to collect answers and contexts...")
    for data_point in eval_data_points:
        question = data_point['question']
        ground_truths = data_point['ground_truths']
        
        eval_data['question'].append(question)
        eval_data['ground_truths'].append(ground_truths)
        eval_data['reference'].append(ground_truths[0])
        
        try:
            response = perform_query(query_engine, question)
            
            generated_answer = str(response)
            eval_data['answer'].append(generated_answer)

            retrieved_contexts = [node.text for node in response.source_nodes]
            eval_data['contexts'].append(retrieved_contexts)
            
            print(f" - Answer: {generated_answer[:100]}...")
            print(f" - Contexts retrieved: {len(retrieved_contexts)}")

        except Exception as e:
            print(f"Error querying RAG pipeline for question '{question}': {e}")
            eval_data['answer'].append("Error: Could not generate answer.")
            eval_data['contexts'].append([])

    df = pd.DataFrame(eval_data)

    dataset = Dataset.from_pandas(df)

    print("\nDataset prepared for RAGas evaluation:")
    print(dataset)

    metrics = [
        #answer_relevancy,
        #faithfulness,
        answer_similarity, 
        context_recall,
        #context_precision,
    ]

    print("\nStarting RAGas evaluation...")
    
    ragas_llm_for_eval = LangchainOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
    ragas_embeddings_for_eval = LangchainOllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    ragas_llm = LangchainLLMWrapper(ragas_llm_for_eval)
    ragas_embeddings = LangchainEmbeddingsWrapper(ragas_embeddings_for_eval)

    score = None
    try:
        score = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
    except TimeoutError as e:
        print(f"\nCaught a TimeoutError during RAGas evaluation: {e}")
        return

    if score:
        print("\n--- RAGas Evaluation Results ---")
        
        scores_df = score.to_pandas()

        final_df = pd.concat([df, scores_df], axis=1)

        print("Overall Scores:")
        for metric in metrics:
            metric_name = metric.name
            mean_score = pd.to_numeric(final_df[metric_name], errors='coerce').mean()
            print(f" - {metric_name.replace('_', ' ').title()}: {mean_score:.4f}")

        print("\nDetailed Scores per sample:")
        print(final_df[['question'] + [metric.name for metric in metrics]].to_markdown(index=False))

if __name__ == "__main__":
    evaluate_rag_system_with_pipeline()