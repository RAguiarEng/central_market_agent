"""Fluxo de trabalho dos agentes no LangGraph
(O gerente de projeto inteligente que coordena uma equipe de especialistas)
Autor: Rodrigo Aguiar
Data: 13/08/2026 (Atualizado: 19/08/2026)
"""

import os
from typing import List, Dict, Any, Optional
from loguru import logger
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from datetime import datetime

from core.state import AgentState
from agents.supervisor import SupervisorAgent
from agents.specialist.base import SpecialistAgent
from config import SPECIALIST_DOCUMENTS

# --- 1. Inicializar Agentes ---
logger.info("Inicializando Agente Supervisor...")
supervisor_agent = SupervisorAgent()

logger.info("Inicializando Agentes Especialistas...")
specialist_agents: Dict[str, SpecialistAgent] = {}
for name in SPECIALIST_DOCUMENTS.keys():
    specialist_agents[name] = SpecialistAgent(name=name)

# --- Definição de Custo por Token ---
COST_PER_TOKEN_SUPERVISOR = 0.000001
COST_PER_TOKEN_SPECIALIST = 0.000001


# --- Função Utilitária para Extração Robusta de Tokens ---
def extract_token_usage(
    ai_message_or_metadata: Any, 
    fallback_text_prompt: str = "", 
    fallback_text_completion: str = ""
) -> Dict[str, int]:
    """
    Extrai tokens de forma resiliente, suportando:
    1. usage_metadata do LangChain ({'input_tokens', 'output_tokens', 'total_tokens'})
    2. response_metadata['token_usage'] (OpenAI standard)
    3. response_metadata['usage'] (OpenRouter standard)
    4. Fallback por estimativa de caracteres (~4 caracteres por token) se o modelo free omitir o header.
    """
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if ai_message_or_metadata is not None:
        # Caso seja um AIMessage do LangChain
        if hasattr(ai_message_or_metadata, "usage_metadata") and ai_message_or_metadata.usage_metadata:
            usage = ai_message_or_metadata.usage_metadata
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        # Caso esteja no response_metadata
        if total_tokens == 0:
            resp_meta = getattr(ai_message_or_metadata, "response_metadata", {})
            if isinstance(ai_message_or_metadata, dict):
                resp_meta = ai_message_or_metadata.get("response_metadata", ai_message_or_metadata)
                usage_meta = ai_message_or_metadata.get("usage_metadata")
                if usage_meta:
                    prompt_tokens = usage_meta.get("input_tokens", 0)
                    completion_tokens = usage_meta.get("output_tokens", 0)
                    total_tokens = usage_meta.get("total_tokens", prompt_tokens + completion_tokens)

            if total_tokens == 0 and isinstance(resp_meta, dict):
                raw_usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
                prompt_tokens = raw_usage.get("prompt_tokens", 0) or raw_usage.get("input_tokens", 0)
                completion_tokens = raw_usage.get("completion_tokens", 0) or raw_usage.get("output_tokens", 0)
                total_tokens = raw_usage.get("total_tokens", prompt_tokens + completion_tokens)

    # Fallback caso a API retorne 0
    if total_tokens == 0 and (fallback_text_prompt or fallback_text_completion):
        prompt_tokens = max(1, len(fallback_text_prompt) // 4) if fallback_text_prompt else 0
        completion_tokens = max(1, len(fallback_text_completion) // 4) if fallback_text_completion else 0
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }


# --- 2. Definir os Nós do Grafo ---

def call_supervisor_router(state: AgentState) -> Dict[str, Any]:
    """
    Nó que chama o Agente Supervisor para rotear a query do usuário.
    Normaliza a decisão do agente e atualiza métricas.
    """
    logger.info("Nó: Chamando o Supervisor para roteamento...")
    user_query = state["user_query"]
    supervisor_start_time = datetime.now().timestamp()

    # Invoca o LLM para obter o AIMessage bruto com os metadados
    llm_response = (supervisor_agent.router_prompt | supervisor_agent.llm).invoke({"user_query": user_query})

    # Parse da resposta para seleção do agente
    parsed_response = supervisor_agent.output_parser.parse(llm_response.content)
    raw_agent = parsed_response.get("next_agent", "geral").strip().lower()

    # Normalização robusta do agente selecionado
    if raw_agent in SPECIALIST_DOCUMENTS:
        next_agent = raw_agent
    else:
        next_agent = "geral"

    # Extrai uso de tokens
    tokens_info = extract_token_usage(
        llm_response,
        fallback_text_prompt=user_query,
        fallback_text_completion=llm_response.content
    )
    supervisor_total_tokens = tokens_info["total_tokens"]
    supervisor_cost = supervisor_total_tokens * COST_PER_TOKEN_SUPERVISOR

    supervisor_end_time = datetime.now().timestamp()

    return {
        "next_agent": next_agent,
        "supervisor_tokens": supervisor_total_tokens,
        "supervisor_cost": supervisor_cost,
        "supervisor_latency": supervisor_end_time - supervisor_start_time,
        "current_agent": "supervisor_router"
    }

def call_specialist_agent(state: AgentState) -> Dict[str, Any]:
    """
    Nó que chama o Agente Especialista selecionado.
    Atualiza o estado com a resposta do especialista e métricas de tokens/custo.
    """
    logger.info(f"Nó: Chamando o Agente Especialista: {state['next_agent']}...")
    agent_name = state["next_agent"]
    user_query = state["user_query"]

    if agent_name not in specialist_agents:
        logger.error(f"Agente especialista '{agent_name}' não encontrado. Retornando erro.")
        return {
            "specialist_response": "Erro: Agente especialista não configurado.", 
            "next_agent": "geral",
            "specialist_tokens": 0,
            "specialist_cost": 0.0,
            "specialist_latency": 0.0,
            "current_agent": agent_name
        }

    specialist_start_time = datetime.now().timestamp()

    # Invoca o especialista
    specialist_response_acp = specialist_agents[agent_name].invoke(user_query, context_id="user_session_123")
    specialist_answer = specialist_response_acp.payload.get("answer", "Não foi possível obter uma resposta do especialista.")

    # Extrai o uso de tokens do payload
    tokens_info = extract_token_usage(
        specialist_response_acp.payload,
        fallback_text_prompt=user_query,
        fallback_text_completion=specialist_answer
    )
    specialist_total_tokens = tokens_info["total_tokens"]
    specialist_cost = specialist_total_tokens * COST_PER_TOKEN_SPECIALIST

    specialist_end_time = datetime.now().timestamp()

    return {
        "specialist_response": specialist_answer,
        "specialist_tokens": specialist_total_tokens,
        "specialist_cost": specialist_cost,
        "specialist_latency": specialist_end_time - specialist_start_time,
        "current_agent": agent_name
    }

def handle_general_query(state: AgentState) -> Dict[str, Any]:
    """
    Nó para lidar com queries roteadas para 'geral'.
    Utiliza o Supervisor para responder e contabiliza os tokens da resposta geral.
    """
    logger.info("Nó: Supervisor gerando resposta para query geral...")
    user_query = state["user_query"]
    start_time = datetime.now().timestamp()
    
    # Chama o supervisor com captura de metadados
    general_result = supervisor_agent.answer_general_query_with_usage(user_query)
    general_answer = general_result.get("answer", "")
    
    # Extrai tokens consumidos na resposta geral
    tokens_info = extract_token_usage(
        general_result,
        fallback_text_prompt=user_query,
        fallback_text_completion=general_answer
    )
    general_tokens = tokens_info["total_tokens"]
    general_cost = general_tokens * COST_PER_TOKEN_SUPERVISOR
    
    end_time = datetime.now().timestamp()

    return {
        "final_answer": general_answer,
        "specialist_tokens": general_tokens,
        "specialist_cost": general_cost,
        "specialist_latency": end_time - start_time,
        "current_agent": "general_handler"
    }

def consolidate_response(state: AgentState) -> Dict[str, Any]:
    """
    Nó para consolidar a resposta do especialista ou a resposta geral em final_answer
    e somar todos os tokens e custos do fluxo.
    """
    logger.info("Nó: Consolidando resposta...")

    final_answer = state.get("specialist_response") or state.get("final_answer") or "Não foi possível gerar uma resposta."

    # Soma os tokens e custos de todas as etapas (supervisor + especialista/geral)
    total_tokens = (state.get("supervisor_tokens") or 0) + (state.get("specialist_tokens") or 0)
    total_cost = (state.get("supervisor_cost") or 0.0) + (state.get("specialist_cost") or 0.0)
    total_latency = (state.get("supervisor_latency") or 0.0) + (state.get("specialist_latency") or 0.0)

    messages = state.get("messages", [])
    messages.append(AIMessage(content=final_answer))

    return {
        "final_answer": final_answer, 
        "messages": messages,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "total_latency": total_latency,
        "current_agent": "consolidator"
    }

# --- 3. Construir o Grafo ---
workflow = StateGraph(AgentState)

workflow.add_node("supervisor_router", call_supervisor_router)

for name in SPECIALIST_DOCUMENTS.keys():
    workflow.add_node(name, lambda s, agent_name=name: call_specialist_agent({**s, "next_agent": agent_name}))

workflow.add_node("general_handler", handle_general_query)
workflow.add_node("consolidator", consolidate_response)

workflow.set_entry_point("supervisor_router")

def route_decision(state: AgentState) -> str:
    """
    Decisão segura de roteamento:
    Se o próximo agente for um especialista válido, vai para o nó correspondente.
    Qualquer outro valor (ex: 'geral', 'general', 'none') direciona para 'general_handler'.
    """
    next_agent = (state.get("next_agent") or "geral").strip().lower()
    if next_agent in SPECIALIST_DOCUMENTS:
        return next_agent
    return "general_handler"

workflow.add_conditional_edges(
    "supervisor_router",
    route_decision,
    {
        "general_handler": "general_handler",
        **{name: name for name in SPECIALIST_DOCUMENTS.keys()}
    }
)

for name in SPECIALIST_DOCUMENTS.keys():
    workflow.add_edge(name, "consolidator")
workflow.add_edge("general_handler", "consolidator")
workflow.add_edge("consolidator", END)

app = workflow.compile()
