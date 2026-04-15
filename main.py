from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os

from singleton.Embedder import EmbeddingModel
from singleton.VectorStore import VectorDB
from singleton.Cerebras import LLM


app = FastAPI(title="Bhagavad Gita RAG API")


# Initialize singletons
embedder = EmbeddingModel(os.environ.get("EMBEDDING_MODEL"))
vector_db = VectorDB("rag/gita_index.faiss", "rag/gita_embeddings.json")
llm = LLM(model_name=os.environ.get("MODEL_NAME"))


# Request/Response models
class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    query: str
    results: List[str]


class ChatRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    message: str
    error: Optional[str] = None


# ---------------------- SEMANTIC SEARCH API ----------------------
@app.post("/search", response_model=SearchResponse)
def semantic_search(request: SearchRequest):
    try:
        query_vector = embedder.get_embedding(request.query).astype("float32")
        results = vector_db.search(query_vector, k=3)
        return SearchResponse(
            query=request.query,
            results=[result["eng_sloka"] for result in results]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"message": "Semantic search failed", "error": str(e)}
        )


# ---------------------- CHAT API ----------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        query_vector = embedder.get_embedding(request.prompt).astype("float32")
        context = [result["verse"] for result in vector_db.search(query_vector, k=3)]

        response = llm.complete(
            user_prompt=request.prompt,
            context=context,
            temperature=request.temperature,
            system_prompt=(
                "You are Lord Krishna, explaining the teachings of the Bhagavad Gita "
                "in a simple, clear, and compassionate manner. Use the provided context "
                "to answer the user's question. Always address the user by their name "
                "in your response. Respond in a natural, human-like way with warmth and empathy. "
                "Avoid robotic phrasing. Use conversational language, gentle guidance, and "
                "relatable analogies when helpful. Your tone should be calm, wise, and reassuring, "
                "like a trusted mentor speaking directly to the user."
            )
        )

        return ChatResponse(message=str(response))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"message": "Chat processing failed", "error": str(e)}
        )
