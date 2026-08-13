""" Carregamento e indexação de documentos em vectorstore para agentes especialistas.
Autor: Rodrigo Aguiar
Data: 12/08/2026
"""

import os

# Bibliotecas externas
from langchain_community.document_loaders import PyPDFLoader            # Usado para carregar PDFs individuais
from langchain_text_splitters import RecursiveCharacterTextSplitter     # Splitter mais robusto
from langchain_ollama import OllamaEmbeddings                           # Modelo de embeddings do Ollama (lembrete: modelos do Cohere são 2ª opção)
from langchain_community.vectorstores import FAISS
from loguru import logger # Para logging estruturado

# Bibliotecas internas (configurações atualizadas)
from config import (
    DOCS_PATH,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SPECIALIST_DOCUMENTS,
    SPECIALIST_FAISS_INDEXES,
)

def load_documents(file_path: str):
    """Carrega um documento PDF específico a partir do caminho fornecido."""
    logger.info(f"Carregando documento: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Documento '{file_path}' carregado. Total de páginas: {len(documents)}")
        return documents
    except Exception as e:
        logger.error(f"Erro ao carregar o documento '{file_path}': {e}")
        raise

def split_documents(documents):
    """Divide os documentos carregados em chunks menores para indexação."""
    logger.info(f"Dividindo documentos em chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Total de chunks criados: {len(chunks)}")
    return chunks

def get_embeddings():
    """Inicializa e retorna o modelo de embeddings do Ollama."""
    logger.info(f"Inicializando modelo de embeddings: {EMBEDDING_MODEL}")
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

def create_and_save_faiss_index(chunks, index_path: str, embeddings: OllamaEmbeddings):
    """Cria um novo índice FAISS a partir dos chunks e o salva localmente."""
    if not chunks:
        logger.warning(f"Nenhum chunk fornecido para criar o índice em {index_path}. Pulando.")
        return None

    logger.info(f"Criando índice FAISS em: {index_path}")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    logger.success(f"Índice FAISS salvo em: {index_path}")
    return vector_store

def load_faiss_index(index_path: str, embeddings: OllamaEmbeddings):
    """Carrega um índice FAISS existente de um caminho local."""
    logger.info(f"Carregando índice FAISS de: {index_path}")
    try:
        # allow_dangerous_deserialization=True é necessário para carregar índices salvos localmente
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Erro ao carregar índice FAISS de '{index_path}': {e}")
        raise

def get_or_create_faiss_index_for_specialist(specialist_name: str):
    """
    Função principal para obter ou criar o índice FAISS para um especialista específico.
    Verifica se o índice já existe. Se sim, carrega-o. Caso contrário, cria-o do zero.
    """
    doc_filename = SPECIALIST_DOCUMENTS.get(specialist_name)
    if not doc_filename:
        logger.error(f"Documento não configurado para o especialista: {specialist_name}. Verifique config.py.")
        raise ValueError(f"Documento não configurado para o especialista: {specialist_name}")

    file_path = os.path.join(DOCS_PATH, doc_filename)
    index_path = SPECIALIST_FAISS_INDEXES[specialist_name]
    embeddings = get_embeddings()

    if os.path.exists(index_path) and os.path.isdir(index_path):
        logger.info(f"Índice FAISS para '{specialist_name}' já existe. Carregando...")
        return load_faiss_index(index_path, embeddings)
    else:
        logger.info(f"Índice FAISS para '{specialist_name}' não encontrado. Criando...")
        documents = load_documents(file_path)
        chunks = split_documents(documents)
        return create_and_save_faiss_index(chunks, index_path, embeddings)

if __name__ == "__main__":
    # Garante que o diretório base para os índices exista
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    logger.info(f"Diretório base para índices FAISS '{FAISS_INDEX_PATH}' verificado/criado.")

    # Itera sobre todos os especialistas definidos em config.py e cria/carrega seus índices
    for specialist_name in SPECIALIST_DOCUMENTS.keys():
        try:
            get_or_create_faiss_index_for_specialist(specialist_name)
        except Exception as e:
            logger.error(f"Falha crítica ao processar o índice para o especialista '{specialist_name}': {e}")
            # Dependendo da criticidade, pode-se decidir parar a execução aqui ou continuar.
            # Por enquanto, apenas se loga o erro e continua com os outros.
    logger.success("Processamento de índices FAISS para todos os especialistas concluído.")
