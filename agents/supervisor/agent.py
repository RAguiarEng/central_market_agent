"""Estrutura do Agente Supervisor com LangGraph
Autor: Rodrigo Aguiar
Data: 13/08/2026 (Atualizado: 18/08/2026)
"""

import os
import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openrouter import ChatOpenRouter
from loguru import logger
from datetime import datetime

from config import LLM_SUPERVISOR, SPECIALIST_DOCUMENTS, OPENROUTER_API_KEY
from .models import AgentSelection

class SupervisorAgent:
    """
    Agente Supervisor responsável por orquestrar a interação entre o usuário
    e os agentes especialistas. Decide o roteamento e responde a perguntas
    gerais e institucionais com base nos resumos e metadados dos manuais.
    """
    def __init__(self):
        self.llm = ChatOpenRouter(model=LLM_SUPERVISOR, api_key=OPENROUTER_API_KEY)

        # --- Carregar os resumos dos especialistas ---
        summaries_path = os.path.join(os.path.dirname(__file__), "specialist_summaries.json")
        try:
            with open(summaries_path, 'r', encoding='utf-8') as f:
                self.specialist_summaries_data = json.load(f)
            # Dicionário indexado pelo nome do agente
            self.specialist_summaries_map = {item['agent']: item for item in self.specialist_summaries_data}
            logger.info("Resumos dos especialistas carregados com sucesso.")
        except FileNotFoundError:
            logger.error(f"Arquivo de resumos '{summaries_path}' não encontrado. O supervisor usará descrições básicas.")
            self.specialist_summaries_data = []
            self.specialist_summaries_map = {}

        self.specialist_names = list(self.specialist_summaries_map.keys())

        # Texto contextual unificado com 100% das informações do JSON
        self.full_context_str = self._format_all_summaries()

        # Cadeia 1: Roteamento
        self.output_parser = JsonOutputParser(pydantic_object=AgentSelection)
        self.router_prompt = self._build_router_prompt()
        self.prompt = self.router_prompt # Alias de compatibilidade retroativa
        self.router_chain = self._build_router_chain()

        # Cadeia 2: Resposta a Dúvidas Gerais / Institucionais
        self.general_qa_prompt = self._build_general_qa_prompt()
        self.general_qa_chain = self._build_general_qa_chain()

        logger.success("Agente Supervisor inicializado com sucesso.")

    def _format_single_summary(self, data: Dict[str, Any]) -> str:
        """Formata todos os campos disponíveis de um especialista no JSON."""
        parts = []
        agent_name = data.get("agent", "desconhecido")
        parts.append(f"### Especialista: '{agent_name}'")
        
        if data.get("description"):
            parts.append(f"- Descrição: {data['description']}")
        if data.get("file"):
            version_str = f" (v{data['version']})" if data.get("version") else ""
            parts.append(f"- Documento Fonte: {data['file']}{version_str}")
        if data.get("department"):
            parts.append(f"- Departamento: {data['department']}")
        if data.get("scope"):
            parts.append(f"- Escopo de Aplicação: {data['scope']}")
        if data.get("classification"):
            parts.append(f"- Classificação: {data['classification']}")
        if data.get("creation_date") or data.get("last_update"):
            dates = []
            if data.get("creation_date"):
                dates.append(f"Criação: {data['creation_date']}")
            if data.get("last_update"):
                dates.append(f"Última Atualização: {data['last_update']}")
            parts.append(f"- Datas: {', '.join(dates)}")
        
        # Trata informações extras (ex: CNPJ, Inscrição Estadual, Endereço)
        more_info = data.get("more_info")
        if more_info:
            if isinstance(more_info, list):
                for item in more_info:
                    if isinstance(item, dict):
                        info_str = ", ".join([f"{k}: {v}" for k, v in item.items()])
                        parts.append(f"- Informações Adicionais: {info_str}")
                    elif str(item).strip():
                        parts.append(f"- Informações Adicionais: {item}")
            elif isinstance(more_info, dict):
                info_str = ", ".join([f"{k}: {v}" for k, v in more_info.items()])
                parts.append(f"- Informações Adicionais: {info_str}")
            elif str(more_info).strip():
                parts.append(f"- Informações Adicionais: {more_info}")

        # Tópicos do sumário
        if data.get("summary") and isinstance(data["summary"], list):
            summary_topics = "; ".join(data["summary"])
            parts.append(f"- Estrutura/Tópicos Cobertos: {summary_topics}")

        return "\n".join(parts)

    def _format_all_summaries(self) -> str:
        """Gera o bloco completo de documentação a partir do JSON."""
        if not self.specialist_summaries_data:
            fallback = []
            for name, doc_filename in SPECIALIST_DOCUMENTS.items():
                fallback.append(f"- '{name}': Especialista no documento '{doc_filename}'.")
            return "\n".join(fallback)

        return "\n\n".join([self._format_single_summary(item) for item in self.specialist_summaries_data])

    def _build_router_prompt(self):
        """Constrói o prompt de decisão de roteamento."""
        system_prompt = f"""Você é o Agente Supervisor do Mercado Central 24h.
Sua principal tarefa é analisar a pergunta do usuário e determinar qual especialista é o mais adequado.

Especialistas Disponíveis e seus Escopos Detalhados:
{self.full_context_str}

Instruções de Decisão:
- Escolha APENAS UM especialista se a dúvida recair claramente sobre o domínio dele.
- Caso a pergunta seja uma saudação, uma dúvida geral sobre a empresa (ex: endereço, CNPJ), uma visão geral de quais manuais existem, ou se nenhum especialista for adequado, responda 'geral'.
- Sua resposta DEVE ser estritamente um objeto JSON conforme as instruções de formato.
"""
        format_instructions = self.output_parser.get_format_instructions()
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_query}\n{format_instructions}"),
        ]).partial(format_instructions=format_instructions)

    def _build_router_chain(self):
        """Cadeia de roteamento estruturada."""
        return (
            {"user_query": RunnablePassthrough()}
            | self.router_prompt
            | self.llm
            | self.output_parser
        )

    def _build_general_qa_prompt(self):
        """Prompt para responder perguntas genéricas e institucionais."""
        system_prompt = f"""Você é o Agente Supervisor e Assistente Institucional do Mercado Central 24h.
Você tem acesso ao catálogo completo de manuais, diretrizes, departamentos e informações institucionais da empresa.

Base de Conhecimento Institucional e Resumo dos Manuais:
{self.full_context_str}

Suas Responsabilidades:
1. Responder cordialmente a saudações e apresentar os serviços do Mercado Central 24h.
2. Responder perguntas institucionais e cadastrais (ex: CNPJ, endereços, escopo das unidades) com base nas informações adicionais disponíveis.
3. Explicar quais manuais e tópicos existem na organização e orientar o usuário sobre o que cada departamento/especialista cobre.
4. Se a pergunta for sobre um assunto que não consta nos manuais da empresa, informe gentilmente que a informação não foi encontrada nos documentos corporativos disponíveis.
5. Mantenha um tom profissional, acolhedor e direto.
"""
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_query}"),
        ])

    def _build_general_qa_chain(self):
        """Cadeia de geração de resposta direta para perguntas genéricas."""
        return (
            {"user_query": RunnablePassthrough()}
            | self.general_qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def route_query(self, user_query: str) -> str:
        """Roteia a query para um especialista ou 'geral'."""
        logger.info(f"Supervisor analisando rota para query: '{user_query}'")
        try:
            parsed = self.router_chain.invoke(user_query)
            if isinstance(parsed, dict):
                selected = parsed.get("next_agent", "geral").strip().lower()
            else:
                selected = str(parsed).strip().lower()
        except Exception as e:
            logger.error(f"Falha ao executar roteamento: {e}. Redirecionando para 'geral'.")
            selected = "geral"

        if selected not in self.specialist_names and selected != "geral":
            logger.warning(f"Especialista '{selected}' inválido. Ajustando para 'geral'.")
            selected = "geral"

        logger.info(f"Resultado do roteamento: '{selected}'")
        return selected

    def answer_general_query(self, user_query: str) -> str:
        """Gera resposta para perguntas genéricas usando o contexto do JSON."""
        logger.info(f"Supervisor respondendo query genérica: '{user_query}'")
        try:
            return self.general_qa_chain.invoke(user_query)
        except Exception as e:
            logger.error(f"Erro ao gerar resposta genérica no supervisor: {e}")
            return "Desculpe, ocorreu um erro ao consultar as informações institucionais. Como posso ajudar?"
    
    def answer_general_query_with_usage(self, user_query: str) -> Dict[str, Any]:
        """Gera resposta para perguntas genéricas e retorna também o objeto AIMessage com os metadados de tokens."""
        logger.info(f"Supervisor respondendo query genérica: '{user_query}'")
        try:
            # Invoca diretamente o prompt + llm (sem StrOutputParser) para manter os metadados
            chain = self.general_qa_prompt | self.llm
            ai_message = chain.invoke({"user_query": user_query})
            return {
                "answer": ai_message.content,
                "usage_metadata": getattr(ai_message, "usage_metadata", None),
                "response_metadata": getattr(ai_message, "response_metadata", {})
            }
        except Exception as e:
            logger.error(f"Erro ao gerar resposta genérica no supervisor: {e}")
            return {
                "answer": "Desculpe, ocorreu um erro ao consultar as informações institucionais. Como posso ajudar?",
                "usage_metadata": None,
                "response_metadata": {}
            }

