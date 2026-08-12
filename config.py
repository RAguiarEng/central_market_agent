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

HUGGIINGFACE_API_KEY: str | None = os.getenv("HF_TOKEN")
LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY")
#PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")

BASE_DIR: Path = pathlib.Path(__file__).parent
CLIENT_FOLDER: str = 'docs/Mercado_Central_24h/'
DOC_PATH: str = str(BASE_DIR / CLIENT_FOLDER)
INDEX_PATH: str = str(BASE_DIR / "faiss_index")

EMBEDDING_MODEL: str = 'bge-m3'     # modelo local
QUERY_MODEL: str = 'gemma3:1b'      # modelo local
LLM_MODEL: str = 'llama3.2'         # modelo local
LLM_EVAL_MODEL: str = 'qwen3:4b'
OPENROUTER_JUDGE_MODEL: str = "nvidia/nemotron-3-super-120b-a12b:free"
CHUNK_SIZE: int = 1250
CHUNK_OVERLAP: int = 125

AVALIAR_RAG: bool = True