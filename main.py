import os
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# ---------------------- LOGGING ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- GLOBALS ----------------------
llm = None
initialization_task = None


# ---------------------- INIT ----------------------
def init_singletons():
    global llm

    try:
        logger.info("Initializing LLM...")

        from singleton.Cerebras import LLM

        model_name = os.environ.get("MODEL_NAME", "llama3.1-8b")
        llm = LLM(model_name=model_name)

        logger.info("LLM initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


# ---------------------- ENSURE INIT ----------------------
async def ensure_initialized():
    global llm, initialization_task

    if llm is not None:
        return True

    if initialization_task is not None and not initialization_task.done():
        try:
            await initialization_task
        except Exception:
            return False

    if llm is None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, init_singletons)
        except Exception:
            return False

    return True


# ---------------------- LIFESPAN ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global initialization_task

    async def _background_init():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, init_singletons)

    initialization_task = asyncio.create_task(_background_init())
    yield


# ---------------------- APP ----------------------
app = FastAPI(
    title="Bhagavad Gita LLM API",
    lifespan=lifespan
)


# ---------------------- MODELS ----------------------
class ChatRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    initialized: bool


# ---------------------- HEALTH ----------------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy" if llm else "degraded",
        initialized=llm is not None
    )


# ---------------------- CHAT ----------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        initialized = await ensure_initialized()
        if not initialized or llm is None:
            raise HTTPException(status_code=503, detail="Service initializing")

        response = llm.complete(
            user_prompt=request.prompt,
            context=[],  # No RAG
            temperature=request.temperature,
            system_prompt="You are Lord Krishna guiding the user."
        )

        return ChatResponse(message=str(response))

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
