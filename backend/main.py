from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import json

# Assuming these imports are from your project
from backend.crew_agents import ContextualRAGCrew
from backend.services.db_connection import setup_pgvector_extension

# --- Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    model: Optional[str] = "gemma2:2b"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0

# --- Global ---
rag_crew_instance: Optional[ContextualRAGCrew] = None

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_crew_instance
    print("🚀 App starting up...", flush=True)
    try:
        setup_pgvector_extension()
        rag_crew_instance = ContextualRAGCrew()
        print("✅ RAG Crew initialized successfully.")
    except Exception as e:
        print(f"❌ Error during RAG Crew initialization: {e}", flush=True)
    yield
    print("🛑 App shutting down.", flush=True)

app = FastAPI(lifespan=lifespan)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Streaming Response Generator (Standard, character-by-character) ---
async def event_generator(response_content: str):
    """
    Generates a standard OpenAI-compatible stream of Server-Sent Events (SSE)
    by simulating a token-by-token (character-by-character) stream.
    """
    print("Streaming: Starting standard event generator...", flush=True)
    # Yield each character as a separate chunk
    for char in response_content:
        chunk = {
            "choices": [{
                "delta": {"content": char},
                "index": 0,
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01) # Small delay for a realistic feel
    
    # After all content is sent, send a final chunk with the stop reason
    final_chunk = {
        "choices": [{
            "delta": {},
            "finish_reason": "stop",
            "index": 0
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    print("Streaming: Sent final 'stop' chunk.", flush=True)

# --- Main Chat Endpoint ---
@app.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])
        model = body.get("model", "gemma2:2b")
        stream = body.get("stream", False)
    except json.JSONDecodeError:
        print("❌ Invalid JSON in request body.", flush=True)
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON in request body."})

    if not messages or not isinstance(messages, list) or not messages[-1].get("content"):
        print("❌ Invalid request format. 'messages' must be a list with content.", flush=True)
        return JSONResponse(status_code=400, content={"detail": "Invalid request format. 'messages' must be a list with at least one item containing 'content'."})

    prompt = messages[-1]["content"]
    print(f"📥 Prompt received: '{prompt}' (stream={stream})", flush=True)

    if rag_crew_instance is None:
        print("❌ RAG Crew not initialized.", flush=True)
        raise HTTPException(status_code=500, detail="RAG Crew not initialized.")

    # This is the crucial part that handles the long-running task
    print("⏳ Starting long-running RAG crew process...", flush=True)
    try:
        response_content_result = await asyncio.to_thread(rag_crew_instance.run_crew, prompt, messages)
        response_content = response_content_result.raw
        print("✅ RAG crew process finished successfully. Time to send response.", flush=True)
    except Exception as e:
        print(f"❌ Error running RAG crew: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error processing prompt: {e}")

    # Handle streaming vs. non-streaming response
    if stream:
        print("➡️ Returning STREAMING response.", flush=True)
        return StreamingResponse(event_generator(response_content), media_type="text/event-stream")
    else:
        print("➡️ Returning NON-STREAMING response.", flush=True)
        return {
            "id": "chatcmpl-rag-001",
            "object": "chat.completion",
            "created": int(datetime.utcnow().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(prompt.split()) + len(response_content.split()),
            }
        }

# --- Other endpoints for Open WebUI ---
@app.get("/")
def root():
    return {"message": "RAG Chatbot API running"}

@app.get("/api/tags")
def get_tags():
    return {
        "models": [
            {
                "name": "gemma2:2b",
                "modified_at": datetime.utcnow().isoformat() + "Z",
                "size": 123456789,
                "digest": "sha256:dummyhash",
                "details": {
                    "family": "gemma",
                    "parameter_size": "2b",
                    "format": "gguf"
                }
            }
        ]
    }

@app.get("/models")
def list_models():
    return {"data": [{"id": "gemma2:2b", "object": "model"}], "object": "list"}

@app.get("/api/version")
def version():
    return {"version": "1.0.0"}

@app.get("/api/ps")
def ps():
    return {"status": "ok"}