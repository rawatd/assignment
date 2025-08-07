import os
from sqlalchemy import create_engine, text
from llama_index.vector_stores.postgres import PGVectorStore
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

def get_db_connection_string():
    """Constructs the database connection string from environment variables."""
    db_name = os.getenv("PG_DATABASE") # Changed from PG_DB_NAME
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT")
    
    if not all([db_name, user, password, host, port]):
        raise ValueError("Database connection variables (PG_DATABASE, PG_USER, PG_PASSWORD, PG_HOST, PG_PORT) must be set in .env")
    
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db_name}"

def setup_pgvector_extension():
    """Ensures the pgvector extension is created in the database."""
    db_connection_string = get_db_connection_string()
    engine = create_engine(db_connection_string)
    
    print("Attempting to connect to DB and ensure PGVector extension...")
    try:
        with engine.connect() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.commit()
        print("PGVector extension ensured/created successfully.")
    except Exception as e:
        print(f"Error connecting to DB or creating PGVector extension: {e}")
        raise # Re-raise the exception to indicate a critical failure

def get_vector_store():
    """Initializes and returns the PGVectorStore instance."""
    db_name = os.getenv("PG_DATABASE") # Changed from PG_DB_NAME
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST")
    port = int(os.getenv("PG_PORT")) # Ensure port is int
    table_name = os.getenv("PG_TABLE_NAME") # Changed from COLLECTION_NAME
    embed_dim = int(os.getenv("EMBEDDING_DIM"))

    if not all([db_name, user, password, host, port, table_name, embed_dim]):
        raise ValueError("All PGVectorStore parameters (PG_DATABASE, PG_USER, PG_PASSWORD, PG_HOST, PG_PORT, PG_TABLE_NAME, EMBEDDING_DIM) must be set in .env")

    print(f"Initializing PGVectorStore for table: {table_name}")
    return PGVectorStore.from_params(
        database=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
        table_name=table_name,
        embed_dim=embed_dim,
    )

if __name__ == "__main__":
    # Example usage:
    try:
        setup_pgvector_extension()
        vector_store = get_vector_store()
        print("PGVectorStore initialized successfully.")
    except Exception as e:
        print(f"Error during DB setup or vector store initialization: {e}")