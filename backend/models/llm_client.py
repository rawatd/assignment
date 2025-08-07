import os
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
from llama_index.llms.litellm import LiteLLM # type: ignore

load_dotenv() # Load environment variables from .env

def initialize_ollama_llm():
    """Initializes and returns the Ollama LLM instance."""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    llm_model_name = os.getenv("LLM_MODEL")

    if not ollama_base_url or not llm_model_name:
        raise ValueError("OLLAMA_BASE_URL and LLM_MODEL must be set in .env")

    print(f"Initializing Ollama LLM: {llm_model_name} from {ollama_base_url}")
    return LiteLLM(model=f"ollama/{llm_model_name}", api_base=ollama_base_url)

if __name__ == "__main__":
    # Example usage:
    try:
        llm = initialize_ollama_llm()
        print("Ollama LLM initialized successfully.")
        # You can add a test call here if needed, e.g., llm.complete("Hello")
    except Exception as e:
        print(f"Error initializing Ollama LLM: {e}")