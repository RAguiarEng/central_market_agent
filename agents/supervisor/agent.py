"""Estrutura do Agente Supervisor com LangGraph
Autor: Rodrigo Aguiar
Data: 13/08/2026
"""

import os
import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama # Manter para referência, mas não será usado
from langchain_openrouter import ChatOpenRouter
from loguru import logger
from datetime import datetime

from config import LLM_MAIN, LLM_SUPERVISOR, SPECIALIST_DOCUMENTS, OPENROUTER_API_KEY # Importa as configurações dos especialistas
from .models import AgentSelection

class SupervisorAgent:
    """
    Agente Supervisor responsável por orquestrar a interação entre o usuário
    e os agentes especialistas. Ele decide qual especialista deve responder
    a uma determinada pergunta do usuário.
    """
    def __init__(self):
        self.llm = ChatOpenRouter(model=LLM_SUPERVISOR, api_key=OPENROUTER_API_KEY)

        # --- Carregar os resumos dos especialistas ---
        summaries_path = os.path.join(os.path.dirname(__file__), "specialist_summaries.json")
        try:
            with open(summaries_path, 'r', encoding='utf-8') as f:
                self.specialist_summaries_data = json.load(f)
            # Converte a lista de dicionários para um dicionário onde a chave é o nome do agente
            self.specialist_summaries_map = {item['agent']: item for item in self.specialist_summaries_data}
            logger.info("Resumo dos especialistas carregados com sucesso.")
        except FileNotFoundError:
            logger.error(f"Arquivo de resumos '{summaries_path}' não encontrado. O supervisor usará descrições básicas.")
            self.specialist_summaries_data = []
            self.specialist_summaries_map = {}

        # Obtém os nomes dos especialistas disponíveis a partir da configuração
        # Uso dos nomes dos agentes do SJON para garantir consistência
        self.specialist_names = list(self.specialist_summaries_map.keys())

        self.output_parser = JsonOutputParser(pydantic_object=AgentSelection)
        self.prompt = self._build_prompt()
        self.router_chain = self._build_router_chain()
        logger.success("Agente Supervisor inicializado com sucesso.")

    def _build_prompt(self):
        """
        Constrói o prompt para o LLM do supervisor, instruindo-o a atuar como um roteador.
        Inclui a descrição dos especialistas disponíveis para auxiliar na decisão.
        """
        specialist_descriptions = []
        if self.specialist_summaries_map: # Verifica se os resumos foram carregados
            for agent_name, data in self.specialist_summaries_map.items():
                # Constrói uma descrição rica para cada especialista
                description_parts = [
                    f"- '{agent_name}': {data.get('description', 'Nenhuma descrição disponível.')}"
                ]
                if data.get('summary'):
                    # Formata a lista de tópicos do sumário
                    formatted_summary = "; ".join(data['summary'])
                    description_parts.append(f"Tópicos principais: {formatted_summary}.")
                if data.get('department'):
                    description_parts.append(f"Departamento responsável: {data['department']}.")
                if data.get('scope'):
                    description_parts.append(f"Escopo: {data['scope']}.")

                specialist_descriptions.append(" ".join(description_parts))
        else:
            # Fallback para a descrição básica se os resumos não forem carregados
            for name, doc_filename in SPECIALIST_DOCUMENTS.items():
                description = f"- '{name}': Especialista no documento '{doc_filename}'. Responde a perguntas sobre {name.replace('_', ' ')}."
                specialist_descriptions.append(description)

        specialist_list_str = "\n".join(specialist_descriptions)

        SYSTEM_PROMPT_CONTENT = f"""Você é um Agente Supervisor inteligente e experiente.
        Sua principal tarefa é analisar a pergunta do usuário e determinar qual dos agentes especialistas disponíveis é o mais adequado para respondê-la.

        Agentes Especialistas Disponíveis:
        {specialist_list_str}

        Instruções:
        - Analise cuidadosamente a pergunta do usuário.
        - Escolha APENAS UM agente especialista que tenha a maior probabilidade de responder à pergunta com base em sua área de especialização.
        - Se nenhum especialista for adequado, ou se a pergunta for genérica demais para ser atribuída a um único especialista, você deve responder 'geral'.
        - Sua resposta DEVE ser um objeto JSON que siga o formato especificado.
        """

        # Obtenha as instruções de formato do parser de saída.
        # Isso garante que o LLM saiba exatamente como formatar o JSON.
        format_instructions = self.output_parser.get_format_instructions()

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_CONTENT),
                ("human", "{user_query}\n{format_instructions}"),
            ]
        ).partial(format_instructions=format_instructions)

        return prompt_template
    

    def _build_router_chain(self):
        """Constrói a cadeia de roteamento para o supervisor."""
        return (
            {"user_query": RunnablePassthrough()}   # A entrada da cadeia é a query do usuário
            | self.prompt                           # Aplica o prompt de roteamento
            | self.llm                              # Invoca o LLM para tomar a decisão
            | self.output_parser                    # Extrai a string da resposta do LLM
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
