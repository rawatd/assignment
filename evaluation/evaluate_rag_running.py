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
from llama_index.llms.ollama import Ollama # LlamaIndex's Ollama LLM
from llama_index.embeddings.ollama import OllamaEmbedding # LlamaIndex's Ollama Embedding

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
# Replace with your actual database connection details or ensure these are set as environment variables
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")
DB_NAME = os.getenv("PG_DATABASE", "uae_db")
DB_USER = os.getenv("PG_USER", "dev")
DB_PASSWORD = os.getenv("PG_PASSWORD", "dev")
DB_TABLE_NAME = os.getenv("PG_TABLE_NAME", "my_uae_hr_laws_embeddings") # The table where your chunks are stored

# --- Ollama LLM and Embedding Configuration (from .env) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma2:2b") # Your LLM model, e.g., "gemma2:2b"

# Using 'all-minilm:33m' as the embedding model with its known dimension of 384
EMBEDDING_MODEL = "all-minilm:33m" 

# --- Configure LlamaIndex global settings ---
# Set your LLM and Embedding model to use LlamaIndex's Ollama
# Increased the request_timeout for the Ollama LLM significantly
Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=3600.0) # Increased to 1 hour
# Corrected: Use 'model_name' instead of 'model' for OllamaEmbedding
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
        # The embed_dim is now explicitly set to 384 for 'all-minilm:33m'
        vector_store = PGVectorStore.from_params(
            database=DB_NAME,
            host=DB_HOST,
            password=DB_PASSWORD,
            port=int(DB_PORT),
            user=DB_USER,
            table_name=DB_TABLE_NAME,
            embed_dim=384,  # Set to 384 for '
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
        print("Also, verify your Ollama server is running and the specified LLM/Embedding models are downloaded.")
        print("Crucially, ensure the 'embed_dim' in PGVectorStore.from_params() matches your embedding model's dimension.")
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
    eval_data_points = [
        {
            'question': "Describe Article (19), Subscription to the Retirement System",
            'ground_truths': [
                "The government authority shall commit to register the citizen employee in the retirement system, and to pay the prescribed subscriptions in accordance with the legislation in force in the Emirate."
            ]
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
            # Use the new fault-tolerant query function
            response = perform_query(query_engine, question)
            
            # Extract the answer
            generated_answer = str(response)
            eval_data['answer'].append(generated_answer)

            # Extract the contexts (source nodes)
            retrieved_contexts = [node.text for node in response.source_nodes]
            eval_data['contexts'].append(retrieved_contexts)
            
            print(f"  - Answer: {generated_answer[:100]}...")
            print(f"  - Contexts retrieved: {len(retrieved_contexts)}")

        except Exception as e:
            print(f"Error querying RAG pipeline for question '{question}': {e}")
            eval_data['answer'].append("Error: Could not generate answer.")
            eval_data['contexts'].append([])

    # Create a pandas DataFrame from the dictionaries
    df = pd.DataFrame(eval_data)

    # Convert the DataFrame to a Hugging Face Dataset object
    dataset = Dataset.from_pandas(df)

    print("\nDataset prepared for RAGas evaluation:")
    print(dataset)

    # Define the RAGas metrics you want to evaluate
    metrics = [
        answer_relevancy,
        faithfulness,
        answer_similarity, 
        context_recall,
        context_precision,
    ]

    print("\nStarting RAGas evaluation...")
    
    # Initialize Langchain's Ollama LLM and Embedding models for RAGas compatibility
    ragas_llm_for_eval = LangchainOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
    ragas_embeddings_for_eval = LangchainOllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    # Wrap Langchain's Ollama LLM and Embedding models for RAGas compatibility
    ragas_llm = LangchainLLMWrapper(ragas_llm_for_eval)
    ragas_embeddings = LangchainEmbeddingsWrapper(ragas_embeddings_for_eval)

    score = None
    try:
        # Evaluate the dataset against the defined metrics
        score = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
    except TimeoutError as e:
        print(f"\nCaught a TimeoutError during RAGas evaluation: {e}")
        print("This often happens when the LLM or Embedding models are taking too long to respond.")
        print("Please check your Ollama server and models. You might need to increase the timeout further.")
        return

    if score:
        print("\n--- RAGas Evaluation Results ---")
        
        # Get the raw scores from the RAGas evaluation
        scores_df = score.to_pandas()

        # Merge the original DataFrame with the scores DataFrame
        # This is a more robust way to ensure all columns are present
        final_df = pd.concat([df, scores_df], axis=1)

        print("Overall Scores:")
        for metric in metrics:
            metric_name = metric.name
            mean_score = pd.to_numeric(final_df[metric_name], errors='coerce').mean()
            print(f"  - {metric_name.replace('_', ' ').title()}: {mean_score:.4f}")

        print("\nDetailed Scores per sample:")
        #columns_to_display = ['question', 'ground_truths', 'answer', 'contexts', 'reference'] + [metric.name for metric in metrics]
        #print(final_df[columns_to_display].to_markdown(index=False))

if __name__ == "__main__":
    evaluate_rag_system_with_pipeline()

