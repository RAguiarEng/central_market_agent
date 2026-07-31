"""Configurações centrais do projeto
Autor: Rodrigo Aguiar
Data: 30/07/2026
"""

# Sistema
import pathlib
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

huggingface_api_key: str | None = os.getenv("HF_TOKEN")
langsmith_api_key: str | None = os.getenv("LANGSMITH_API_KEY")
#pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY")

BASE_DIR: Path = pathlib.Path(__file__).parent
CLIENT_FOLDER: str = 'docs/Mercado_Central_24h/'
DOC_PATH: str = str(BASE_DIR / CLIENT_FOLDER)
INDEX_PATH: str = str(BASE_DIR / "faiss_index")

EMBEDDING_MODEL: str = 'bge-m3'
QUERY_MODEL: str = 'gemma3:1b'
LLM_MODEL: str = 'llama3.2'
LLM_EVAL_MODEL: str = 'qwen3:4b'
CHUNK_SIZE: int = 1250
CHUNK_OVERLAP: int = 125

AVALIAR_RAG: bool = True