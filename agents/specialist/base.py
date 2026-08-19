""" Coração da lógica de cada agente especialista.
Autor: Rodrigo Aguiar
Data: 12/08/2026 (Atualizado: 18/08/2026)
"""

import os
import time
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openrouter import ChatOpenRouter
from loguru import logger
from datetime import datetime
from langchain_core.documents import Document

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

    def invoke(self, query: str, context_id: str) -> ACPMessage:
        """
        Invoca o agente especialista com uma query e retorna uma mensagem ACP.
        Para depuração e métricas, o payload incluirá os documentos e os metadados de tokens.
        """
        logger.info(f"Agente '{self.name}' recebendo query para context_id '{context_id}': '{query}'")
        retrieved_docs_content: List[str] = []
        try:
            start_time = time.time()

            # 1. Recuperar documentos relevantes usando o retriever
            retrieved_docs: List[Document] = self.retriever.invoke(query)

            logger.debug(f"Agente '{self.name}' recuperou {len(retrieved_docs)} documentos para a query '{query}'.")
            retrieved_docs_content = [doc.page_content for doc in retrieved_docs]

            # 2. Formatar o contexto para o LLM
            context_for_llm = "\n\n".join(retrieved_docs_content)

            # 3. Invocar o LLM diretamente para preservar os metadados (AIMessage)
            chain = self.prompt | self.llm
            ai_message = chain.invoke({"context": context_for_llm, "question": query})

            response_content = ai_message.content

            logger.info(f"Agente '{self.name}' gerou resposta para context_id '{context_id}'.")

            # 4. Incluir resposta, documentos e metadados de uso no payload
            payload = {
                "answer": response_content,
                "query": query,
                "retrieved_docs": retrieved_docs_content,
                "usage_metadata": getattr(ai_message, "usage_metadata", None),
                "response_metadata": getattr(ai_message, "response_metadata", {})
            }

            confidence = 0.8
            sources = ["documento:" + self.name]

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
