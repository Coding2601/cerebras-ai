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
embedder = None
vector_db = None
llm = None
initialization_task = None

def get_base_path():
    """Get the base path for the application (works for both local and deployment)."""
    return Path(__file__).parent.resolve()

def init_singletons():
    """Initialize singletons lazily with error handling."""
    global embedder, vector_db, llm
    
    if embedder is not None:
        return
    
    try:
        logger.info("Initializing singletons...")
        
        from singleton.Embedder import EmbeddingModel
        from singleton.VectorStore import VectorDB
        from singleton.Cerebras import LLM
        
        base_path = get_base_path()
        
        # Initialize embedder
        embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model: {embedding_model}")
        embedder = EmbeddingModel(embedding_model)
        
        # Initialize vector DB with absolute paths
        faiss_path = base_path / "rag" / "gita_index.faiss"
        json_path = base_path / "rag" / "gita_embeddings.json"
        
        logger.info(f"Loading FAISS index from: {faiss_path}")
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata not found: {json_path}")
            
        vector_db = VectorDB(str(faiss_path), str(json_path))
        
        # Initialize LLM
        model_name = os.environ.get("MODEL_NAME", "llama3.1-8b")
        logger.info(f"Loading LLM: {model_name}")
        llm = LLM(model_name=model_name)
        
        logger.info("Singletons initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize singletons: {e}")
        raise

async def ensure_initialized():
    """Ensure singletons are initialized, waiting for background task if needed."""
    global embedder, vector_db, llm, initialization_task
    
    if embedder is not None:
        return True
    
    # If background task is running, wait for it
    if initialization_task is not None and not initialization_task.done():
        logger.info("Waiting for background initialization...")
        try:
            await initialization_task
        except Exception as e:
            logger.error(f"Background initialization failed: {e}")
            return False
    
    # If still not initialized, do it now
    if embedder is None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, init_singletons)
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager - start server immediately, init in background."""
    global initialization_task
    
    logger.info("Starting up...")
    # Start initialization in background so server can bind port immediately
    async def _background_init():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, init_singletons)
    
    initialization_task = asyncio.create_task(_background_init())
    logger.info("Server started, models loading in background...")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="Bhagavad Gita RAG API",
    lifespan=lifespan
)


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


class HealthResponse(BaseModel):
    status: str
    initialized: bool


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy" if all([embedder, vector_db, llm]) else "degraded",
        initialized=all([embedder, vector_db, llm])
    )


# ---------------------- SEMANTIC SEARCH API ----------------------
@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    try:
        initialized = await ensure_initialized()
        if not initialized or embedder is None or vector_db is None:
            raise HTTPException(status_code=503, detail="Service initializing, please retry")
            
        query_vector = embedder.get_embedding(request.query).astype("float32")
        results = vector_db.search(query_vector, k=3)
        return SearchResponse(
            query=request.query,
            results=[result["eng_sloka"] for result in results]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Semantic search failed", "error": str(e)}
        )


# ---------------------- CHAT API ----------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        initialized = await ensure_initialized()
        if not initialized or embedder is None or vector_db is None or llm is None:
            raise HTTPException(status_code=503, detail="Service initializing, please retry")
            
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Chat processing failed", "error": str(e)}
        )

