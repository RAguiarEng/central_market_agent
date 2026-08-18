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

# --- Definição de Custo por Token (Simplificado para Exemplo) ---
# Estes valores são apenas exemplos e podem variar muito entre modelos e provedores.
# Para um cálculo preciso, você precisaria consultar a tabela de preços do OpenRouter/Ollama.
# Assumimos um custo médio para o supervisor e para os especialistas.
COST_PER_TOKEN_SUPERVISOR = 0.000001 # Ex: $1 por milhão de tokens
COST_PER_TOKEN_SPECIALIST = 0.0000005 # Ex: $0.50 por milhão de tokens (se for um modelo menor/local)


# --- 2. Definir os Nós do Grafo ---

def call_supervisor_router(state: AgentState) -> Dict[str, Any]:
    """
    Nó que chama o Agente Supervisor para rotear a query do usuário.
    Atualiza o estado com o nome do próximo agente.
    """
    logger.info("Nó: Chamando o Supervisor para roteamento...")
    user_query = state["user_query"]

    # Captura o tempo de início da execução do supervisor
    supervisor_start_time = datetime.now().timestamp()

    # Invoca a parte do LLM da cadeia para obter a resposta bruta (AIMessage)
    # que contém o response_metadata. 
    llm_response = (supervisor_agent.router_prompt | supervisor_agent.llm).invoke({"user_query": user_query})

    # Usa o output_parser do supervisor para parsear o conteúdo da resposta do LLM 
    # no modelo Pydantic AgentSelection.
    parsed_response = supervisor_agent.output_parser.parse(llm_response.content)

    # Acessa o atributo 'next_agent' diretamente do objeto Pydantic parseado.
    next_agent = parsed_response.get("next_agent", "geral").strip().lower()

    # Extrai o uso de tokens do response_metadata da resposta bruta do LLM.
    token_usage = llm_response.response_metadata.get("token_usage", {})
    supervisor_prompt_tokens = token_usage.get("prompt_tokens", 0)
    supervisor_completion_tokens = token_usage.get("completion_tokens", 0)
    supervisor_total_tokens = token_usage.get("total_tokens", 0)

    supervisor_cost = supervisor_total_tokens * COST_PER_TOKEN_SUPERVISOR

    # Captura o tempo de fim da execução do supervisor
    supervisor_end_time = datetime.now().timestamp()

    return {
        "next_agent": next_agent,
        "supervisor_tokens": supervisor_total_tokens,
        "supervisor_cost": supervisor_cost,
        "supervisor_latency": supervisor_end_time - supervisor_start_time,
        "current_agent": "supervisor_router" # Indica que o supervisor acabou de rodar
    }

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
        return {"specialist_response": "Erro: Agente especialista não configurado.", 
                "next_agent": "geral",          # Redireciona para geral se o agente não for encontrado
                "specialist_tokens": 0,
                "specialist_cost": 0.0,
                "specialist_latency": 0.0,
                "current_agent": agent_name     # Indica qual agente tentou rodar
        }

    # Captura o tempo de início da execução do especialista
    specialist_start_time = datetime.now().timestamp()

    # Invoca o especialista e captura a resposta completa para acessar metadados
    specialist_response_acp = specialist_agents[agent_name].invoke(user_query, context_id="user_session_123")

    # Extrai a resposta do payload do ACPMessage
    specialist_answer = specialist_response_acp.payload.get("answer", "Não foi possível obter uma resposta do especialista.")

    # Extrai o uso de tokens do especialista
    # Nota: A estrutura de response_metadata pode variar. Para Ollama, pode estar em specialist_response_acp.payload.get("response_metadata")
    token_usage = specialist_response_acp.payload.get("response_metadata", {}).get("token_usage", {})
    specialist_prompt_tokens = token_usage.get("prompt_tokens", 0)
    specialist_completion_tokens = token_usage.get("completion_tokens", 0)
    specialist_total_tokens = token_usage.get("total_tokens", 0)

    specialist_cost = specialist_total_tokens * COST_PER_TOKEN_SPECIALIST

    # Captura o tempo de fim da execução do especialista
    specialist_end_time = datetime.now().timestamp()

    return {
        "specialist_response": specialist_answer,
        "specialist_tokens": specialist_total_tokens,
        "specialist_cost": specialist_cost,
        "specialist_latency": specialist_end_time - specialist_start_time,
        "current_agent": agent_name         # Indica qual agente acabou de rodar
    }

def handle_general_query(state: AgentState) -> Dict[str, Any]:
    """
    Nó para lidar com queries roteadas para 'geral'.
    Utiliza o Supervisor para responder usando os dados do specialist_summaries.json.
    """
    logger.info("Nó: Supervisor gerando resposta para query geral...")
    user_query = state["user_query"]
    
    start_time = datetime.now().timestamp()
    
    # Chama a nova cadeia do supervisor
    general_answer = supervisor_agent.answer_general_query(user_query)
    
    end_time = datetime.now().timestamp()

    return {
        "final_answer": general_answer,
        "specialist_tokens": 0,
        "specialist_cost": 0.0,
        "specialist_latency": end_time - start_time,
        "current_agent": "general_handler"
    }


def consolidate_response(state: AgentState) -> Dict[str, Any]:
    """
    Nó para consolidar a resposta do especialista ou a resposta geral em final_answer.
    """
    logger.info("Nó: Consolidando resposta...")

    final_answer = state.get("specialist_response") or state.get("final_answer") or "Não foi possível gerar uma resposta."

    # Soma os tokens e custos de todas as etapas (supervisor + especialista/geral)
    total_tokens = state.get("supervisor_tokens", 0) + state.get("specialist_tokens", 0)
    total_cost = state.get("supervisor_cost", 0.0) + state.get("specialist_cost", 0.0)
    total_latency = state.get("supervisor_latency", 0.0) + state.get("specialist_latency", 0.0)

    # Adiciona a resposta final ao histórico de mensagens
    messages = state.get("messages", [])
    messages.append(AIMessage(content=final_answer))

    return {
        "final_answer": final_answer, 
        "messages": messages,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "total_latency": total_latency,
        "current_agent": "consolidator"     # Indica que o consolidator acabou de rodar
    }


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
        "general_handler": "general_handler",                       # Se o supervisor disser 'geral', vai para o handler geral
        **{name: name for name in SPECIALIST_DOCUMENTS.keys()}      # Para cada especialista, vai para o nó correspondente
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
        initial_state = AgentState(
            user_query=query, 
            final_answer=None, 
            messages=[HumanMessage(content=query)], 
            next_agent=None, 
            specialist_response=None,
            # Inicializa métricas para o LangGraph somar
            supervisor_tokens=0, supervisor_cost=0.0, supervisor_latency=0.0,
            specialist_tokens=0, specialist_cost=0.0, specialist_latency=0.0,
            total_tokens=0, total_cost=0.0, total_latency=0.0,
            current_agent=None, error_message=None, context_id=None,
            trace_id=None, start_time=None, end_time=None, recursion_limit_counter=None
        )

        # Executar o grafo
        final_state = app.invoke(initial_state)

        logger.info(f"Resposta Final para '{query}': {final_state['final_answer']}")
        logger.info(f"Histórico de Mensagens: {final_state['messages']}")
        logger.info(f"Métricas: Tokens={final_state['total_tokens']}, Custo=${final_state['total_cost']:.6f}, Latência={final_state['total_latency']:.2f}s")
        print("=" * 80)

    logger.info("Teste do grafo LangGraph concluído.")
