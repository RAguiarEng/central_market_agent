"""Fluxo de trabalho dos agentes no LangGraph
(O gerente de projeto inteligente que coordena uma equipe de especialistas)
Autor: Rodrigo Aguiar
Data: 13/08/2026
"""

import os
from typing import List, Dict, Any
from loguru import logger
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

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

# --- 2. Definir os Nós do Grafo ---

def call_supervisor_router(state: AgentState) -> Dict[str, Any]:
    """
    Nó que chama o Agente Supervisor para rotear a query do usuário.
    Atualiza o estado com o nome do próximo agente.
    """
    logger.info("Nó: Chamando o Supervisor para roteamento...")
    user_query = state["user_query"]
    next_agent = supervisor_agent.route_query(user_query)
    return {"next_agent": next_agent}

def call_specialist_agent(state: AgentState) -> Dict[str, Any]:
    """
    Nó que chama o Agente Especialista selecionado.
    Atualiza o estado com a resposta do especialista.
    """
    logger.info(f"Nó: Chamando o Agente Especialista: {state['next_agent']}...")
    agent_name = state["next_agent"]
    user_query = state["user_query"]

    if agent_name not in specialist_agents:
        logger.error(f"Agente especialista '{agent_name}' não encontrado. Retornando erro.")
        return {"specialist_response": "Erro: Agente especialista não configurado.", "next_agent": "geral"}

    specialist_response_acp = specialist_agents[agent_name].invoke(user_query, context_id="user_session_123")

    # Extrai a resposta do payload do ACPMessage
    specialist_answer = specialist_response_acp.payload.get("answer", "Não foi possível obter uma resposta do especialista.")

    return {"specialist_response": specialist_answer}

def handle_general_query(state: AgentState) -> Dict[str, Any]:
    """
    Nó para lidar com queries que o supervisor roteou para 'geral'.
    Pode ser um LLM genérico ou uma mensagem padrão.
    """
    logger.info("Nó: Lidando com query geral...")
    user_query = state["user_query"]
    # Por enquanto, uma resposta padrão. Futuramente, pode ser um LLM genérico.
    general_answer = f"Desculpe, não consegui encontrar um especialista específico para '{user_query}'. Por favor, tente reformular sua pergunta."
    return {"final_answer": general_answer}

def consolidate_response(state: AgentState) -> Dict[str, Any]:
    """
    Nó para consolidar a resposta do especialista ou a resposta geral em final_answer.
    """
    logger.info("Nó: Consolidando resposta...")
    if state.get("specialist_response"):
        final_answer = state["specialist_response"]
    elif state.get("final_answer"): # Se já houver uma resposta geral
        final_answer = state["final_answer"]
    else:
        final_answer = "Não foi possível gerar uma resposta."

    # Adiciona a resposta final ao histórico de mensagens
    messages = state.get("messages", [])
    messages.append(AIMessage(content=final_answer))

    return {"final_answer": final_answer, "messages": messages}


# --- 3. Construir o Grafo ---
workflow = StateGraph(AgentState)

# Adicionar o nó inicial (Supervisor Router)
workflow.add_node("supervisor_router", call_supervisor_router)

# Adicionar nós para cada especialista
for name in SPECIALIST_DOCUMENTS.keys():
    # Usamos uma função lambda para criar um closure e passar o nome do agente
    workflow.add_node(name, lambda s, agent_name=name: call_specialist_agent({**s, "next_agent": agent_name}))

# Adicionar nó para queries gerais
workflow.add_node("general_handler", handle_general_query)

# Adicionar nó para consolidar a resposta
workflow.add_node("consolidator", consolidate_response)

# Definir o ponto de entrada
workflow.set_entry_point("supervisor_router")

# Definir as arestas condicionais (roteamento do supervisor)
# O supervisor_router decide para onde ir em seguida
def route_decision(state: AgentState) -> str:
    next_agent = state["next_agent"]
    if next_agent == "geral":
        return "general_handler"
    else:
        return next_agent # Roteia para o nó do especialista com o mesmo nome

workflow.add_conditional_edges(
    "supervisor_router",
    route_decision,
    {
        "general_handler": "general_handler", # Se o supervisor disser 'geral', vai para o handler geral
        **{name: name for name in SPECIALIST_DOCUMENTS.keys()} # Para cada especialista, vai para o nó correspondente
    }
)

# Após o especialista ou o handler geral, consolidar a resposta
for name in SPECIALIST_DOCUMENTS.keys():
    workflow.add_edge(name, "consolidator")
workflow.add_edge("general_handler", "consolidator")

# O consolidator é o ponto final do fluxo
workflow.add_edge("consolidator", END)

# Compilar o grafo
app = workflow.compile()

# --- 4. Testar o Grafo ---
if __name__ == "__main__":
    logger.info("Iniciando teste do grafo LangGraph...")

    test_queries = [
        "Qual o horário de funcionamento do mercado?",
        "Como um novo fornecedor pode se cadastrar?",
        "Quero saber sobre a política de trocas.",
        "Quais são os procedimentos operacionais internos?",
        "Qual a política de devolução de produtos eletrônicos?",
        "Olá, tudo bem?",
        "Me fale sobre o Mercado Central 24h."
    ]

    for query in test_queries:
        logger.info(f"\n--- Processando Query: '{query}' ---")
        # O estado inicial do grafo
        initial_state = AgentState(user_query=query, final_answer=None, messages=[HumanMessage(content=query)], next_agent=None, specialist_response=None)

        # Executar o grafo
        final_state = app.invoke(initial_state)

        logger.info(f"Resposta Final para '{query}': {final_state['final_answer']}")
        logger.info(f"Histórico de Mensagens: {final_state['messages']}")
        print("=" * 80)

    logger.info("Teste do grafo LangGraph concluído.")
