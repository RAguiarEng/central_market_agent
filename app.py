"""Interface Streamlit - Mercado Central 24h
Autor: Rodrigo Aguiar (https://raguiar.eng.br)
Data: 13/08/2026
"""

import streamlit as st
from loguru import logger
from datetime import datetime
import time

from rag_multiagent import app as langgraph_app
from core.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

# --- Configurações da Página Streamlit ---
st.set_page_config(
    page_title="Mercado Central 24h - Assistente Inteligente",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Customização de Estilos (UI/UX Custom CSS) ---
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

    /* Fontes Globais */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
    }

    /* Container Principal */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Hero Banner Header */
    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
    }
    
    .hero-badges {
        display: flex;
        gap: 10px;
        margin-bottom: 12px;
        flex-wrap: wrap;
    }

    .badge-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .badge-tech {
        display: inline-flex;
        align-items: center;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .hero-title {
        font-size: 1.95rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f8fafc, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.96rem;
        line-height: 1.55;
        margin: 0;
    }

    /* Cards de Métricas da Sidebar */
    .metric-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: 'Outfit', sans-serif;
    }

    .agent-pill {
        display: inline-block;
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Barra Lateral - Footer de Créditos */
    .sidebar-footer {
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
    }

    .sidebar-footer p {
        font-size: 0.82rem;
        color: #94a3b8;
        margin: 0;
    }

    .sidebar-footer a {
        color: #38bdf8;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .sidebar-footer a:hover {
        color: #0284c7;
        text-decoration: underline;
    }

    /* Estilização das Mensagens do Chat */
    .stChatMessage {
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Botão Primário / Limpar Chat */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.2s ease;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Hero Banner: Título e Descrição da Empresa ---
st.markdown("""
<div class="hero-card">
    <div class="hero-badges">
        <span class="badge-status">● Operação 24/7</span>
        <span class="badge-tech">⚡ LangGraph Multi-Agent RAG</span>
        <span class="badge-tech">⭐ Clube Cliente VIP Central</span>
    </div>
    <h1 class="hero-title">🛒 Mercado Central 24h</h1>
    <p class="hero-subtitle">
        Supermercado moderno com operação contínua, integração física-digital, delivery e app próprio. 
        Converse em tempo real com nossos agentes especialistas em <strong>gestão de estoque</strong>, 
        <strong>políticas de atendimento</strong> e <strong>fidelidade VIP</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Área Principal: Chat com o Usuário ---
# Inicializa o histórico de chat na sessão do Streamlit, se ainda não existir
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Olá! Bem-vindo ao **Mercado Central 24h**. Como posso ajudar você hoje com produtos, estoque ou programa VIP?"
    })

# Exibe as mensagens anteriores do chat com avatares customizados
for message in st.session_state.messages:
    avatar_icon = "🛒" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# Campo de entrada para o usuário
user_query = st.chat_input("Digite sua dúvida sobre produtos, pedidos, horário ou fidelidade...")

if user_query:
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🛒"):
        with st.spinner("Consultando agentes especialistas..."):
            # --- Lógica de Integração com o Grafo LangGraph  ---
            initial_state = AgentState(
                user_query=user_query, 
                final_answer=None, 
                messages=[HumanMessage(content=user_query)],
                next_agent=None, 
                specialist_response=None,
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

            total_latency = final_state.get("total_latency", 0.0)
            total_tokens = final_state.get("total_tokens", 0)
            total_cost = final_state.get("total_cost", 0.0)
            selected_agent = final_state.get("next_agent", "N/A")
            supervisor_latency = final_state.get("supervisor_latency", 0.0)
            specialist_latency = final_state.get("specialist_latency", 0.0)
            trace_id = final_state.get("trace_id", "N/A")

            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})

            # --- Atualiza Métricas na Sessão ---
            st.session_state.last_request_metrics = {
                "total_latency": total_latency,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "selected_agent": selected_agent,
                "supervisor_latency": supervisor_latency,
                "specialist_latency": specialist_latency,
                "trace_id": trace_id
            }

# --- Barra Lateral (Métricas e Controles) ---
with st.sidebar:
    st.markdown("### 📊 Painel de Métricas")
    st.caption("Estatísticas da última inferência executada pelo sistema.")

    if "last_request_metrics" in st.session_state:
        metrics = st.session_state.last_request_metrics
        
        # Grid de métricas em 2 colunas
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">⏱️ Latência Total</div>
                <div class="metric-value">{metrics['total_latency']:.2f}s</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">💲 Custo Est.</div>
                <div class="metric-value">${metrics['total_cost']:.5f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">🎟️ Total Tokens</div>
                <div class="metric-value">{metrics['total_tokens']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">🤖 Agente</div>
                <span class="agent-pill">{metrics['selected_agent']}</span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔍 Detalhes Técnicos & Latência"):
            st.write(f"**Supervisor Latência:** `{metrics['supervisor_latency']:.2f}s`")
            st.write(f"**Especialista Latência:** `{metrics['specialist_latency']:.2f}s`")
            st.write(f"**Trace ID:** `{metrics['trace_id']}`")
    else:
        st.info("💡 Envie uma mensagem para visualizar as métricas detalhadas em tempo real.")

    st.markdown("---")
    
    # Botão de Ação: Limpar Chat
    if st.button("🗑️ Limpar Histórico de Chat", use_container_width=True):
        st.session_state.messages = []
        if "last_request_metrics" in st.session_state:
            del st.session_state.last_request_metrics
        st.rerun()

    # --- Rodapé da Barra Lateral ---
    st.markdown("""
    <div class="sidebar-footer">
        <p>Criado por <a href="https://raguiar.eng.br" target="_blank">Rodrigo Aguiar</a></p>
    </div>
    """, unsafe_allow_html=True)

logger.info("Aplicativo Streamlit iniciado.")
