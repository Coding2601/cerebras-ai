# Bhagavad Gita RAG API

A FastAPI-based RAG (Retrieval-Augmented Generation) application for querying the Bhagavad Gita. The API provides semantic search and conversational responses powered by Cerebras LLM.

## Features

- **Semantic Search**: Find relevant Gita verses using vector similarity search
- **Chat API**: Conversational AI that answers questions as Lord Krishna using retrieved context
- **Singleton Pattern**: Efficient resource management for embedding models, vector store, and LLM

## Tech Stack

- **FastAPI** - Web framework
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Text embeddings
- **Cerebras Cloud SDK** - LLM inference
- **Pydantic** - Data validation

## Setup

### Prerequisites

- Python 3.9+
- Cerebras API key

### Installation

1. Clone the repository and navigate to the project:
```bash
cd cerebras-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```env
CEREBRAS_API_KEY=your_cerebras_api_key
EMBEDDING_MODEL=all-MiniLM-L6-v2
MODEL_NAME=llama3.1-8b
```

### Run the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /search

Performs semantic search on Bhagavad Gita verses.

**Request:**
```json
{
  "query": "What does Krishna say about duty?"
}
```

**Response:**
```json
{
  "query": "What does Krishna say about duty?",
  "results": [
    "English translation of sloka 1...",
    "English translation of sloka 2...",
    "English translation of sloka 3..."
  ]
}
```

### POST /chat

Chat with Lord Krishna persona using RAG.

**Request:**
```json
{
  "prompt": "How should I face challenges in life?",
  "temperature": 0.7
}
```

**Response:**
```json
{
  "message": "My dear friend, life presents many challenges..."
}
```

## Project Structure

```
.
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── singleton/             # Singleton classes
│   ├── Cerebras.py        # LLM client
│   ├── Embedder.py        # Embedding model
│   └── VectorStore.py     # FAISS vector database
├── rag/                   # Data files
│   ├── gita_index.faiss   # FAISS index
│   └── gita_embeddings.json # Verse metadata
└── .env                   # Environment variables
```

## Architecture

The application uses **singleton pattern** for resource management:

- `EmbeddingModel`: Loads sentence transformer model once
- `VectorDB`: Loads FAISS index and metadata once
- `LLM`: Initializes Cerebras client once

This ensures efficient memory usage and fast request handling.

## API Documentation

When the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CEREBRAS_API_KEY` | Cerebras API key | `your_api_key_here` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `MODEL_NAME` | Cerebras model name | `llama3.1-8b` |

## License

MIT
