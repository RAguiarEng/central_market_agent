""" Carregamento e indexação de documentos em vectorstore
Autor: Rodrigo Aguiar
Data: 30/07/2026
"""

# Sistema
import os

# Bibliotecas externas
from langchain_community.document_loaders import DirectoryLoader
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Bibliotecas internas
from config import *

# Obs.: Princípio central de tipagem duck-typing: 
# não importa a classe, desde que implemente os métodos necessários.

def carregar_ou_criar_vectorstore(embeddings: Embeddings) -> FAISS:
    """Reaproveita o índice FAISS salvo em disco, se existir."""
    if os.path.exists(INDEX_PATH):
        print("Carregando índice FAISS existente...")
        return FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )

    print("Índice não encontrado. Processando documentos...")
    pdfs = DirectoryLoader(DOC_PATH, glob='*.pdf', loader_kwargs={"languages": ["por"]}).load()
    # Usando "por" de acordo com a convenção ISO 639-2/T
    print(f"Qde de PDFs carregados: {len(pdfs)}")
    
    tokenizer = AutoTokenizer.from_pretrained(f'BAAI/{EMBEDDING_MODEL}')
    splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    pedacos = splitter.split_documents(pdfs)
    print(f'Qde de pedaços criados: {len(pedacos)}')

    vectorstore = FAISS.from_documents(pedacos, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"Índice salvo em: {INDEX_PATH}")
    return vectorstore