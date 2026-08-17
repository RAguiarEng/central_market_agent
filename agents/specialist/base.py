""" Coração da lógica de cada agente especialista.
Autor: Rodrigo Aguiar
Data: 12/08/2026
"""

# RAG/agents/specialist/base.py

import os
import time
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
#from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter
from loguru import logger
from datetime import datetime
from langchain_core.documents import Document # Importar Document para tipagem

from core.protocols import ACPMessage
from indexing import get_or_create_faiss_index_for_specialist
from config import LLM_MAIN, TOP_K_RETRIEVAL

class SpecialistAgent:
    """
    Classe base para um Agente Especialista.
    Cada especialista é responsável por um documento específico,
    possui seu próprio índice FAISS e gera respostas sobre ele.
    """
    def __init__(self, name: str):
        self.name = name
        logger.info(f"Inicializando Agente Especialista '{self.name}'...")
        self.vector_store = get_or_create_faiss_index_for_specialist(name)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
        self.llm = ChatOpenRouter(model=LLM_MAIN)
        self.prompt = self._build_prompt()
        # A cadeia de geração será construída e invocada dentro do método 'invoke'
        logger.success(f"Agente Especialista '{self.name}' inicializado com sucesso.")

    def _build_prompt(self):
        """Constrói o prompt para o LLM do especialista."""
        template = """Você é um agente especialista em {specialist_name} do Mercado Central 24H.
        Sua tarefa é responder à pergunta do usuário utilizando o contexto fornecido.
        Se a resposta não estiver diretamente no contexto, tente inferir a informação mais provável com base no que foi fornecido.
        Se, mesmo com inferência, a resposta não puder ser razoavelmente construída a partir do contexto, diga que não sabe.
        Não invente informações que não tenham qualquer base no contexto.

        Contexto:
        {context}

        Pergunta: {question}

        Resposta:"""
        return ChatPromptTemplate.from_template(template).partial(specialist_name=self.name)

    def _build_rag_generation_chain(self):
        """
        Constrói a cadeia de geração RAG que combina o prompt com o LLM.
        Esta cadeia espera um dicionário com 'context' (string) e 'question' (string).
        """
        return (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, query: str, context_id: str) -> ACPMessage:
        """
        Invoca o agente especialista com uma query e retorna uma mensagem ACP.
        Para depuração, o payload incluirá os documentos recuperados.
        """
        logger.info(f"Agente '{self.name}' recebendo query para context_id '{context_id}': '{query}'")
        retrieved_docs_content: List[str] = []
        try:
            start_time = time.time() # Inicia contagem de tempo

            # 1. Recuperar documentos relevantes usando o retriever
            retrieved_docs: List[Document] = self.retriever.invoke(query)

            # --- LOG de recuperação de documento
            logger.debug(f"Agente ´{self.name}' recuperou {len(retrieved_docs)} documentos para a query '{query}'.")
            for i, doc in enumerate(retrieved_docs):
                logger.debug(f"  Documento {i+1} (Fonte: {doc.metadata.get('source', 'N/A', )}): {doc.page_content[:500]}...")  # loga os 500 primeiros caracteres

            retrieved_docs_content = [doc.page_content for doc in retrieved_docs]

            # 2. Formatar o contexto para o LLM (uma única string com todos os chunks)
            context_for_llm = "\n\n".join(retrieved_docs_content)

            # 3. Invocar a cadeia de geração RAG com o contexto e a pergunta
            generation_chain = self._build_rag_generation_chain()
            response_content = generation_chain.invoke({"context": context_for_llm, "question": query})

            logger.info(f"Agente '{self.name}' gerou resposta para context_id '{context_id}'.")

            # Incluir os documentos recuperados no payload para depuração
            payload = {
                "answer": response_content,
                "query": query,
                "retrieved_docs": retrieved_docs_content # Adiciona o conteúdo dos documentos recuperados
            }

            # TODO: Implementar cálculo de confiança e extração de fontes reais
            confidence = 0.8 # Placeholder
            sources = ["documento:" + self.name] # Placeholder

            return ACPMessage(
                sender=self.name,
                receiver="supervisor",
                intent="final_answer",
                payload=payload,
                context_id=context_id,
                timestamp=datetime.now().isoformat() + "Z",
                confidence=confidence,
                sources=sources
            )
        except Exception as e:
            logger.error(f"Erro no agente '{self.name}' para context_id '{context_id}': {e}")
            return ACPMessage(
                sender=self.name,
                receiver="supervisor",
                intent="error",
                payload={"error": str(e), "query": query, "retrieved_docs": retrieved_docs_content},
                context_id=context_id,
                timestamp=datetime.now().isoformat() + "Z",
                confidence=0.0
            )
