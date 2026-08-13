"""Estrutura do Agente Supervisor com LangGraph
Autor: Rodrigo Aguiar
Data: 13/07/2026
"""

import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama # Manter para referência, mas não será usado
from langchain_openrouter import ChatOpenRouter
from loguru import logger
from datetime import datetime

from config import LLM_MAIN, LLM_SUPERVISOR, SPECIALIST_DOCUMENTS, OPENROUTER_API_KEY # Importa as configurações dos especialistas

class SupervisorAgent:
    """
    Agente Supervisor responsável por orquestrar a interação entre o usuário
    e os agentes especialistas. Ele decide qual especialista deve responder
    a uma determinada pergunta do usuário.
    """
    def __init__(self):
        self.llm = ChatOpenRouter(model=LLM_SUPERVISOR, api_key=OPENROUTER_API_KEY)
        # Obtém os nomes dos especialistas disponíveis a partir da configuração
        self.specialist_names = list(SPECIALIST_DOCUMENTS.keys())
        self.prompt = self._build_prompt()
        self.router_chain = self._build_router_chain()
        logger.success("Agente Supervisor inicializado com sucesso.")

    def _build_prompt(self):
        """
        Constrói o prompt para o LLM do supervisor, instruindo-o a atuar como um roteador.
        Inclui a descrição dos especialistas disponíveis para auxiliar na decisão.
        """
        specialist_descriptions = []
        for name, doc_filename in SPECIALIST_DOCUMENTS.items():
            # Cria uma descrição mais detalhada para cada especialista
            # O nome do arquivo PDF pode dar uma pista sobre o conteúdo
            description = f"- '{name}': Especialista no documento '{doc_filename}'. Responde a perguntas sobre {name.replace('_', ' ')}."
            specialist_descriptions.append(description)

        specialist_list_str = "\n".join(specialist_descriptions)

        template = f"""Você é um Agente Supervisor inteligente e experiente.
        Sua principal tarefa é analisar a pergunta do usuário e determinar qual dos agentes especialistas disponíveis é o mais adequado para respondê-la.

        Agentes Especialistas Disponíveis:
        {specialist_list_str}

        Instruções:
        - Analise cuidadosamente a pergunta do usuário.
        - Escolha APENAS UM agente especialista que tenha a maior probabilidade de responder à pergunta com base em sua área de especialização.
        - Se nenhum especialista for adequado, ou se a pergunta for genérica demais para ser atribuída a um único especialista, você deve responder 'geral'.
        - Sua resposta deve ser APENAS o nome do agente especialista escolhido (ou 'geral'). Não adicione explicações ou qualquer outro texto.

        Exemplos:
        Pergunta: "Qual o horário de funcionamento?"
        Resposta: faq_clientes_funcionarios

        Pergunta: "Como posso me cadastrar como fornecedor?"
        Resposta: manual_fornecedores_compras

        Pergunta: "Quero saber sobre a política de trocas."
        Resposta: politica_atendimento_trocas

        Pergunta: "Quais são os procedimentos operacionais internos?"
        Resposta: regulamento_interno_operacional

        Pergunta: "Olá, tudo bem?"
        Resposta: geral

        Pergunta: {{user_query}}
        Resposta:"""
        return ChatPromptTemplate.from_template(template)

    def _build_router_chain(self):
        """Constrói a cadeia de roteamento para o supervisor."""
        return (
            {"user_query": RunnablePassthrough()}   # A entrada da cadeia é a query do usuário
            | self.prompt                           # Aplica o prompt de roteamento
            | self.llm                              # Invoca o LLM para tomar a decisão
            | StrOutputParser()                     # Extrai a string da resposta do LLM
        )

    def route_query(self, user_query: str) -> str:
        """
        Analisa a query do usuário e retorna o nome do especialista selecionado
        ou 'geral' se nenhum especialista for adequado.
        """
        logger.info(f"Supervisor recebendo query para roteamento: '{user_query}'")
        # Invoca a cadeia de roteamento e normaliza a saída
        selected_specialist = self.router_chain.invoke(user_query).strip().lower()

        # Validação para garantir que o LLM não invente nomes de especialistas
        if selected_specialist not in self.specialist_names and selected_specialist != "geral":
            logger.warning(f"Supervisor selecionou um especialista inválido: '{selected_specialist}'. Redirecionando para 'geral'.")
            return "geral"

        logger.info(f"Supervisor roteou a query para: '{selected_specialist}'")
        return selected_specialist
