"""Configurações centrais do projeto
Autor: Rodrigo Aguiar
Data: 30/07/2026
"""

# Sistema
import pathlib
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configurações de LangSmith ---
LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT")

# --- Configurações de Modelos ---
# Modelo de embeddings para vetorização de documentos e queries
EMBEDDING_MODEL: str = 'bge-m3'     # modelo local (2ª opção: modelos do Cohere)
# Modelo principal de LLM para síntese de respostas pelos agentes
LLM_MAIN: str = 'llama3.2'          # modelo local (antes era o LLM_MODEL)
# Modelo de LLM leve para reescrita de query (pode ser usado pelo supervisor futuramente)
LLM_REWRITE: str = 'gemma3:1b'      # modelo local (antes era o QUERY_MODEL)

# --- Configurações de RAG ---
# Tamanho dos pedaços de texto ao dividir os documentos
CHUNK_SIZE: int = 500
# Tamanho da sobreposição entre os pedaçoes para manter contexto
CHUNK_OVERLAP: int = 100
# Qde de pedaços a serem recuperados por busca
TOP_K_RETRIEVAL: int = 5

# --- Caminhos de Diretórios ---
# Caminho basae para os documentos PDF
DOCS_PATH: str = "docs/Mercado_Central_24h"
# Caminho base para salvar os índices FAISS
FAISS_INDEX_PATH: str = "faiss_index"

# --- Configurações dos Agentes Especialistas --- 
# Mapeia o nome descritivo de cada especialista ao nome do arquivo PDF que ele gerencia.
SPECIALIST_DOCUMENTS: dict = {
    "faq_clientes_funcionarios": "FAQ_Clientes_Funcionarios.pdf",
    "manual_fornecedores_compras": "Manual_Fornecedores_Politica_Compras.pdf",
    "politica_atendimento_trocas": "Politica_Atendimento_Trocas_Devolucoes.pdf",
    "regulamento_interno_operacional": "Regulamento_Interno_Procedimentos_Operacionais.pdf",
}

# Gera automaticamente os caminhos completos para os diretórios dos índices FAISS de cada especialista.
SPECIALIST_FAISS_INDEXES: dict = {
    name: os.path.join(FAISS_INDEX_PATH, f"{name}_faiss_index")
    for name in SPECIALIST_DOCUMENTS.keys()
}

# Informações antigas:
#BASE_DIR: Path = pathlib.Path(__file__).parent
#CLIENT_FOLDER: str = 'docs/Mercado_Central_24h/'
#DOC_PATH: str = str(BASE_DIR / CLIENT_FOLDER)
#INDEX_PATH: str = str(BASE_DIR / "faiss_index") 
#HUGGIINGFACE_API_KEY: str | None = os.getenv("HF_TOKEN")
#PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")
#OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
#LLM_EVAL_MODEL: str = 'qwen3:4b'
#OPENROUTER_JUDGE_MODEL: str = "nvidia/nemotron-3-super-120b-a12b:free"
#AVALIAR_RAG: bool = True