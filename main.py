import os
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for singletons (initialized lazily)
# embedder = None
vector_db = None
llm = None
initialization_task = None


def get_base_path():
    return Path(__file__).parent.resolve()


def init_singletons():
    global vector_db, llm  # removed embedder
    
    try:
        logger.info("Initializing singletons...")
        
        # from singleton.Embedder import EmbeddingModel
        from singleton.VectorStore import VectorDB
        from singleton.Cerebras import LLM
        
        base_path = get_base_path()
        
        # -------- EMBEDDER DISABLED --------
        # embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        # logger.info(f"Loading embedding model: {embedding_model}")
        # embedder = EmbeddingModel(embedding_model)
        
        # Initialize vector DB
        faiss_path = base_path / "rag" / "gita_index.faiss"
        json_path = base_path / "rag" / "gita_embeddings.json"
        
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata not found: {json_path}")
            
        vector_db = VectorDB(str(faiss_path), str(json_path))
        
        # Initialize LLM
        model_name = os.environ.get("MODEL_NAME", "llama3.1-8b")
        llm = LLM(model_name=model_name)
        
        logger.info("Singletons initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize singletons: {e}")
        raise


async def ensure_initialized():
    global vector_db, llm, initialization_task
    
    if vector_db is not None:
        return True
    
    if initialization_task is not None and not initialization_task.done():
        try:
            await initialization_task
        except Exception:
            return False
    
    if vector_db is None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, init_singletons)
        except Exception:
            return False
    
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global initialization_task
    
    async def _background_init():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, init_singletons)
    
    initialization_task = asyncio.create_task(_background_init())
    yield


app = FastAPI(
    title="Bhagavad Gita RAG API",
    lifespan=lifespan
)


# Models
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


class HealthResponse(BaseModel):
    status: str
    initialized: bool


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy" if all([vector_db, llm]) else "degraded",
        initialized=all([vector_db, llm])
    )


# ---------------------- SEARCH API (DISABLED EMBEDDING) ----------------------
@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    raise HTTPException(
        status_code=501,
        detail="Semantic search disabled (embedder commented out)"
    )


# ---------------------- CHAT API ----------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        initialized = await ensure_initialized()
        if not initialized or vector_db is None or llm is None:
            raise HTTPException(status_code=503, detail="Service initializing")

        # -------- EMBEDDING DISABLED --------
        # query_vector = embedder.get_embedding(request.prompt).astype("float32")
        # context = [result["verse"] for result in vector_db.search(query_vector, k=3)]

        context = []  # fallback: no retrieval

        response = llm.complete(
            user_prompt=request.prompt,
            context=context,
            temperature=request.temperature,
            system_prompt="You are Lord Krishna guiding the user."
        )

        return ChatResponse(message=str(response))

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
