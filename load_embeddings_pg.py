import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.text_splitter import SentenceSplitter

# --- Load Environment Variables ---
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DATABASE = os.getenv("PG_DATABASE")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_TABLE_NAME = os.getenv("PG_TABLE_NAME", "markdown_docs")

EMBEDDING_DIM = 384  # for sentence-transformers/all-MiniLM-L6-v2

# --- Set Up LLM and Embedding Model ---
Settings.llm = Ollama(model="gemma:2b", request_timeout=3600.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Set Up PGVector ---
pg_vector_store = PGVectorStore.from_params(
    database=PG_DATABASE,
    host=PG_HOST,
    password=PG_PASSWORD,
    port=int(PG_PORT),
    user=PG_USER,
    table_name=PG_TABLE_NAME,
    embed_dim=EMBEDDING_DIM,
)

# --- Load and Split Markdown Documents ---
documents = SimpleDirectoryReader(input_dir="output_markdown", required_exts=[".md"]).load_data()

# Use SentenceSplitter directly to chunk into nodes
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
nodes = splitter.get_nodes_from_documents(documents)

# --- Store Embeddings into Postgres ---
storage_context = StorageContext.from_defaults(vector_store=pg_vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

print(f"[✔] Successfully embedded and stored {len(nodes)} chunks into PGVector.")
