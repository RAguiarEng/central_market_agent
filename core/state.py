"""Definição do AgentState para o LangGraph
Autor: Rodrigo Aguiar
Data: 30/07/2026
"""

from typing import List, TypedDict, Annotated, Dict, Any
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from operator import add

# Definição do estado do grafo como um TypedDict para 
# compatibilidade com LangGraph e para ter tipagem estática.
class AgentState(TypedDict):
    """
    Representa o estado global da conversa que é passado entre os nós do LangGraph.
    Este estado reflete a memória hierárquica: o supervisor guarda o suficiente
    para decidir e interagir, enquanto os especialistas contribuem com detalhes.
    """
    user_query: str                                         # A pergunta original do usuário.
    current_agent: str                                      # O agente que está processando a requisição no momento.
    conversation_history: Annotated[List[BaseMessage], add] # Histórico completo da conversa, incluindo mensagens do usuário e do sistema.
    agent_responses: Annotated[List[Dict[str, Any]], add]   # Respostas parciais ou finais dos agentes especialistas.
    selected_specialist: str                                # O nome do especialista selecionado pelo supervisor para a query atual.
    final_answer: str                                       # A resposta final consolidada pelo supervisor.
    error_message: str                                      # Mensagens de erro ou falha no processamento.
    context_id: str                                         # ID da sessão para rastreamento.

    # Campos para avaliação e observabilidade:
    trace_id: str           # ID do trace no LangSmith para rastreamento completo.
    start_time: float       # Timestamp de início da execução.
    end_time: float         # Timestamp de fim da execução.
    total_tokens: int       # Contagem total de tokens utilizados.
    total_cost: float       # Custo estimado da execução.

    # Campo para controle de loops e falhas
    recursion_limit_counter: int        # Contador para evitar loops infinitos.
