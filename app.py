"""Interface Streamlit
Autor: Rodrigo Aguiar
Data: 13/08/2026
"""

import streamlit as st
from loguru import logger
from datetime import datetime
import time # Para medir o tempo decorrido

from rag_multiagent import app as langgraph_app
from core.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

# --- Configurações da Página Streamlit ---
st.set_page_config(
    page_title="Mercado Central 24h - Assistente Inteligente",  # Título da aba do navegador
    page_icon="🛒",                                             # Ícone na aba do navegador
    layout="wide",                                              # Usa a largura total da tela
    initial_sidebar_state="expanded"                            # Barra lateral expandida por padrão
)

# --- Título e Descrição da Empresa ---
st.title("🛒 Mercado Central 24h - Seu Assistente Inteligente")

company_description = """
**Supermercado moderno de operação contínua (24/7)** que integra a experiência de loja física com serviços de delivery e aplicativo próprio.

Nosso foco principal é a **eficiência operacional na gestão de estoque** e uma **forte política de atendimento ao cliente**, 
impulsionada pelo seu programa de fidelidade "Cliente VIP Central".
"""
st.markdown(company_description)

st.markdown("---") # Separador visual para melhor organização

# --- Área Principal: Chat com o Usuário ---
st.header("Converse com nosso Assistente!")

# Inicializa o histórico de chat na sessão do Streamlit, se ainda não existir
# st.session_state é como um dicionário que persiste entre as reruns do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Olá! Como posso ajudar você hoje no Mercado Central 24h?"})


# Exibe as mensagens anteriores do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # 'user' ou 'assistant'
        st.markdown(message["content"])

# Campo de entrada para o usuário
user_query = st.chat_input("Digite sua pergunta aqui...")

if user_query:
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."): # Mostra um spinner enquanto o assistente processa
            # --- Lógica de Integração com o Grafo LangGraph  ---
            # Prepara o estado inicial para o grafo
            initial_state = AgentState(
                user_query=user_query, 
                final_answer=None, 
                messages=[HumanMessage(content=user_query)], # Passa a query do usuário como HumanMessage
                next_agent=None, 
                specialist_response=None,
                # Inicializa métricas para o LangGraph somar
                supervisor_tokens=0, supervisor_cost=0.0, supervisor_latency=0.0,
                specialist_tokens=0, specialist_cost=0.0, specialist_latency=0.0,
                total_tokens=0, total_cost=0.0, total_latency=0.0,
                current_agent=None, error_message=None, context_id=None,
                trace_id=None, start_time=None, end_time=None, recursion_limit_counter=None
            )

            # Invoca o grafo LangGraph
            final_state = langgraph_app.invoke(initial_state)

            # Extrai a resposta final e as métricas do estado final
            response_content = final_state.get("final_answer", "Desculpe, não consegui gerar uma resposta.")

            # Métricas para exibição
            total_latency = final_state.get("total_latency", 0.0)
            total_tokens = final_state.get("total_tokens", 0)
            total_cost = final_state.get("total_cost", 0.0)
            selected_agent = final_state.get("next_agent", "N/A")
            supervisor_latency = final_state.get("supervisor_latency", 0.0)
            specialist_latency = final_state.get("specialist_latency", 0.0)
            trace_id = final_state.get("trace_id", "N/A")

            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})

            # --- Atualiza e Exibe Métricas na Barra Lateral ---
            st.session_state.last_request_metrics = {
                "total_latency": total_latency,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "selected_agent": selected_agent,
                "supervisor_latency": supervisor_latency,
                "specialist_latency": specialist_latency,
                "trace_id": trace_id
            }

# --- Barra Lateral para Métricas ---
st.sidebar.title("📊 Métricas da Última Requisição")

# Exibe as métricas se existirem na sessão
if "last_request_metrics" in st.session_state:
    metrics = st.session_state.last_request_metrics
    st.sidebar.write(f"**Tempo Decorrido:** {metrics['total_latency']:.2f} segundos")
    st.sidebar.write(f"**Tokens Consumidos:** {metrics['total_tokens']}")
    st.sidebar.write(f"**Custo Estimado:** ${metrics['total_cost']:.6f}") # Mais casas decimais para custo
    st.sidebar.write(f"**Agente Roteado:** {metrics['selected_agent']}")
    st.sidebar.write(f"**ID da Requisição:** {metrics['trace_id']}")
else:
    st.sidebar.info("As métricas detalhadas de cada requisição aparecerão aqui após sua interação com o assistente.")

# Adicionar um botão para limpar o histórico do chat
if st.sidebar.button("Limpar Chat"):
    st.session_state.messages = []
    if "last_request_metrics" in st.session_state:
        del st.session_state.last_request_metrics # Limpa as métricas também
    st.rerun() # Recarrega a página para refletir as mudanças

logger.info("Aplicativo Streamlit iniciado.")
