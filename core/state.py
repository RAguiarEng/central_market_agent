"""Definição do AgentState para o LangGraph
Autor: Rodrigo Aguiar
Data: 30/07/2026
"""

from typing import List, TypedDict, Annotated, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
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
    current_agent: Optional[str]                            # O agente que está processando a requisição no momento.
    messages: Annotated[List[BaseMessage], add_messages]    # Histórico de mensagens para manter o contexto da conversa.
    agent_responses: List[Dict[str, Any]]                   # Respostas parciais ou finais dos agentes especialistas.
    next_agent: Optional[str]                               # Nome do especialista selecionado pelo supervisor
    final_answer: Optional[str]                             # A resposta final consolidada pelo supervisor.
    specialist_response: Optional[str]                      # Resposta do especialista (se houver)
    error_message: Optional[str]                            # Mensagens de erro ou falha no processamento.
    context_id: Optional[str]                               # ID da sessão para rastreamento.

    # Campos para avaliação e observabilidade (opcional no MVP)
    trace_id: Optional[str]           # ID do trace no LangSmith para rastreamento completo.
    start_time: Optional[float]       # Timestamp de início da execução.
    end_time: Optional[float]         # Timestamp de fim da execução.
    total_tokens: Optional[int]       # Contagem total de tokens utilizados.
    total_cost: Optional[float]       # Custo estimado da execução.

    # Campo para controle de loops e falhas (opcional no MVP)
    recursion_limit_counter: Optional[int]        # Contador para evitar loops infinitos.
