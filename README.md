
##  Project Goal

To build an **interactive chatbot** interface using **Open WebUI** that integrates a **Contextual Retrieval-Augmented Generation (RAG)** pipeline powered by a locally hosted LLM and vector database. The system leverages structured knowledge from user-provided **PDF documents** to deliver high-quality contextual responses.

---

##  Technical Stack

| Tool/Technology       | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| **Docling**           | PDF to Markdown conversion (document pre-processing)                    |
| **LlamaIndex**        | Indexing and RAG framework for document querying                        |
| **PGVector + PostgreSQL** | Vector storage and similarity search                                  |
| **Ollama (LLAMA-3 8B)** | Local LLM hosting and embedding generation                              |
| **Embedding Model**   | `all-MiniLM-L6-v2` for text vectorization (384-dimensional embeddings)  |
| **Crew.AI**           | Agentic orchestration framework for building multi-agent applications   |
| **Arize Phoenix**     | Observability and tracing for LLM pipelines                             |
| **RAGAs**             | RAG pipeline evaluation using metrics like precision, recall, faithfulness |
| **Open WebUI**        | Frontend UI for chat interactions with LLMs                             |
| **FastAPI**           | Exposes backend APIs to be consumed by Open WebUI                       |

---

##  What is Contextual RAG?

**Contextual RAG** enhances the traditional Retrieval-Augmented Generation pipeline by **maintaining and evolving conversational context** over time.

> It ensures not just relevant retrieval, but **alignment between the answer, the retrieved context, and the user’s intent** in a multi-turn dialogue.

This makes it ideal for building interactive agents that require nuanced understanding across turns, such as customer support bots, internal knowledge agents, or domain-specific advisors.

---

##  Component Breakdown

### 📄 Docling: Document Pre-Processor
Converts input PDF documents (with text, tables, and images) into a **clean Markdown** format for downstream processing.

---

###  LlamaIndex + PGVector/PostgreSQL

- **LlamaIndex**: Framework for indexing and querying unstructured data with LLMs.
- **PGVector**: PostgreSQL extension enabling efficient **vector similarity search**.
- Together, they form the backbone of the **RAG** pipeline.

---

###  Contextual RAG (Anthropic-style)

Combines:
- **Embedding model**: Converts documents and queries into vector format.
- **LLM**: Generates contextual answers.
- **Re-ranking**: Ensures the most relevant chunks are selected based on the evolving dialogue context.

---

###  Ollama + LLAMA-3 8B

- **Ollama**: Lightweight platform to run LLMs locally.
- **LLAMA-3 8B**: The primary reasoning model for response generation.
- **MiniLM (all-MiniLM-L6-v2)**: Embedding model for vector generation.

---

###  Crew.AI: Agentic Framework

Used to create **multi-agent task flows**, enabling different AI agents to collaborate and fulfill complex user intents through modular tasks.

---

###  Arize Phoenix: Observability Layer

Helps visualize:
- Prompt inputs/outputs
- Token usage
- Latency
- RAG chain traces

---

###  RAGAs: Evaluation Framework

Evaluates the performance of the RAG pipeline using:
- **Precision**
- **Recall**
- **Faithfulness**
- **Answer correctness**

Helps in **benchmarking** and **iteratively improving** the chatbot’s output quality.

---

###  Open WebUI: Chat Interface

Frontend interface that allows users to interact with the chatbot in a **clean and user-friendly** environment, supporting multi-turn conversations and context preservation.

---

###  FastAPI: Backend API

Serves as the **bridge between Open WebUI and the chatbot backend**, providing endpoints for:
- Chat completion
- Tag/model listing
- Query handling
- Agent pipeline orchestration

---


## Architecture Diagram and Flow

### **Enhanced Text-Based Architecture Diagram**

To visualize the flow, I have enclosed the components in boxes and added **Sequence Flow Numbers (①, ②, ③...)** to show how a user’s question travels through the system.

```text
+------------------------------------------------------------------------------------------+
|                                  USER INTERFACE                                          |
|  ① [ Open WebUI (Docker) ] <-------------------------------------------+                |
+---------|---------------------------------------------------------------|----------------+
          | (User Query)                                                  | (Final Response)
          v                                                               |
+-----------------------+           +-------------------------------------|----------------+
|      API LAYER        |           |          AGENTIC ORCHESTRATION      |                |
|  ② [ FastAPI ]        | --------> |  ③ [ Crew.AI (Specialist Agents) ] -+                |
+-----------------------+           +---------|--------------------------------------------+
                                              | (Task Delegation)
                                              v
+------------------------------------------------------------------------------------------+
|                                    RAG ENGINE                                            |
|  ④ [ LlamaIndex + Contextual RAG Logic ]                                                 |
+---------|-----------------------------------|--------------------------------------------+
          | (Vector Retrieval)                | (Augmentation & LLM Call)
          v                                   v
+-------------------------+         +--------------------------+     +---------------------+
|    VECTOR DATABASE      |         |     LOCAL LLM ENGINE     |     |    OBSERVABILITY    |
| ⑤ [ PGVector / PG ]     |         | ⑥ [ Ollama (Gemma 2b) ]  |     | ⑦ [ MLflow / Phoenix]|
+-------------------------+         +--------------------------+     |     [ RAGAs Eval ]  |
                                                                     +---------------------+

```

---

### **Sequential Step-by-Step Flow**

1. **① User Input:** The user types a question regarding HR or Procurement into the **Open WebUI**.
2. **② API Routing:** The request is sent to **FastAPI**, which acts as the gateway to your Python logic.
3. **③ Agent Orchestration:** **Crew.AI** receives the request. A "Researcher" agent is assigned to find facts using the LlamaIndex tool.
4. **④ Contextual Retrieval:** **LlamaIndex** doesn't just look for keywords; it looks for the **Contextualized Chunks** (the ones we prepared with document summaries).
5. **⑤ Database Query:** LlamaIndex performs a similarity search in **PGVector** to find the top N most relevant chunks.
6. **⑥ Generation:** The retrieved context is sent to **Ollama (Gemma 2b)**. The LLM synthesizes the final answer based *only* on the provided context.
7. **⑦ Monitoring & Eval:** Throughout this process, **MLflow** records the trace, **Arize Phoenix** monitors the prompt quality, and **RAGAs** calculates the faithfulness score.

---

### **The "Contextual Ingestion" Pre-Process (Running Before Chat)**

Before the user can chat, the data must be prepared. This is a one-time setup flow:

1. **Docling** converts the 5 PDFs into Clean Markdown.
2. **Gemma (Ollama)** reads each document and generates a "Situation Summary" for every chunk.
3. **LlamaIndex** combines (Chunk + Summary), creates an embedding, and saves it into **PGVector**.


