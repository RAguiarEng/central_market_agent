"""Interface Streamlit - Mercado Central 24h
Autor: Rodrigo Aguiar (https://raguiar.eng.br)
Data: 13/08/2026
"""

import uuid
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
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

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
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
        backdrop-filter: blur(8px);
    }

    .metric-box:hover {
        border-color: rgba(56, 189, 248, 0.25);
        background: rgba(30, 41, 59, 0.75);
    }
    
    .metric-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
        display: flex;
        align-items: baseline;
        gap: 4px;
    }

    .metric-unit {
        font-size: 0.8rem;
        font-weight: 500;
        color: #64748b;
    }

    .agent-badge-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 4px;
    }

    .agent-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .cost-highlight {
        color: #34d399;
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
    }

    .metric-subtext {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Detalhes Técnicos dentro do Expander */
    .tech-detail-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .tech-detail-key {
        font-size: 0.76rem;
        color: #94a3b8;
        font-weight: 500;
    }

    .tech-detail-val {
        font-size: 0.8rem;
        font-family: monospace;
        color: #e2e8f0;
        font-weight: 600;
    }

    /* Empty State na Sidebar */
    .sidebar-empty-state {
        background: rgba(30, 41, 59, 0.3);
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        margin-bottom: 15px;
    }

    .sidebar-empty-state i {
        font-size: 1.5rem;
        color: #64748b;
        margin-bottom: 8px;
    }

    .sidebar-empty-state p {
        font-size: 0.82rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.4;
    }

    /* Barra Lateral - Footer de Créditos e Redes */
    .sidebar-footer {
        margin-top: 1.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
    }

    .sidebar-footer .author-title {
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0 0 8px 0;
    }

    .sidebar-footer .author-title a {
        color: #38bdf8;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .sidebar-footer .author-title a:hover {
        color: #0284c7;
        text-decoration: underline;
    }

    .sidebar-social-links {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }

    .sidebar-social-links a {
        color: #94a3b8;
        font-size: 1.15rem;
        transition: all 0.25s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .sidebar-social-links a:hover {
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }

    .sidebar-footer .location-text {
        font-size: 0.72rem;
        color: #64748b;
        margin: 0;
        line-height: 1.4;
    }

    /* Estilização das Mensagens do Chat */
    .stChatMessage {
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Botão Limpar Chat */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.86rem;
        border: 1px solid rgba(239, 68, 68, 0.25);
        background: rgba(239, 68, 68, 0.08);
        color: #fca5a5;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: rgba(239, 68, 68, 0.18);
        border-color: rgba(239, 68, 68, 0.45);
        color: #fee2e2;
        transform: translateY(-1px);
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
            # Gera um identificador único de rastreamento para a requisição
            request_trace_id = str(uuid.uuid4())
            
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
                trace_id=request_trace_id, start_time=None, end_time=None, recursion_limit_counter=None
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
    st.caption("Estatísticas da última inferência executada pelo sistema multi-agente.")

    if "last_request_metrics" in st.session_state:
        metrics = st.session_state.last_request_metrics
        
        # Formata o nome do agente para exibição amigável
        raw_agent = metrics.get('selected_agent', 'N/A')
        formatted_agent = raw_agent.replace('_', ' ').title() if raw_agent != "N/A" else "Não Atribuído"

        # 1. Card "Agente" isoladamente no topo
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label"><i class="fa-solid fa-robot"></i> Agente Roteado</div>
            <div class="agent-badge-container">
                <span class="agent-pill">
                    <i class="fa-solid fa-microchip"></i> {formatted_agent}
                </span>
                <span style="font-size: 0.72rem; color: #34d399; font-weight: 600;">● Ativo</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Próxima linha: "Latência total" e "Total tokens" alinhados na mesma linha
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label"><i class="fa-solid fa-stopwatch"></i> Latência Total</div>
                <div class="metric-value">
                    {metrics['total_latency']:.2f}<span class="metric-unit">s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label"><i class="fa-solid fa-ticket"></i> Total Tokens</div>
                <div class="metric-value">
                    {metrics['total_tokens']:,}<span class="metric-unit">tok</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 3. Próxima linha: Card "Custo Est." sozinho nessa linha com legenda
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label"><i class="fa-solid fa-dollar-sign"></i> Custo Estimado</div>
            <div class="cost-highlight">${metrics['total_cost']:.5f}</div>
            <div class="metric-subtext">
                <i class="fa-solid fa-circle-info"></i> Base: $1.00 / 1M de tokens
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4. Próxima linha: Bloco "Detalhes Técnicos & Latência"
        with st.expander("🔍 Detalhes Técnicos & Latência"):
            sup_lat = metrics.get('supervisor_latency') or 0.0
            esp_lat = metrics.get('specialist_latency') or 0.0
            raw_trace = metrics.get('trace_id')
            trace_val = str(raw_trace) if raw_trace else "N/A"
            trace_display = f"{trace_val[:12]}..." if trace_val != "N/A" and len(trace_val) > 12 else trace_val
            st.markdown(f"""
            <div class="tech-detail-card">
                <span class="tech-detail-key">Supervisor Latência</span>
                <span class="tech-detail-val">{sup_lat:.2f}s</span>
            </div>
            <div class="tech-detail-card">
                <span class="tech-detail-key">Especialista Latência</span>
                <span class="tech-detail-val">{esp_lat:.2f}s</span>
            </div>
            <div class="tech-detail-card">
                <span class="tech-detail-key">Trace ID</span>
                <span class="tech-detail-val">{trace_display}</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"ID Completo: `{trace_val}`")
    else:
        st.markdown("""
        <div class="sidebar-empty-state">
            <i class="fa-solid fa-chart-simple"></i>
            <p>Envie uma mensagem para visualizar a telemetria e o custo em tempo real.</p>
        </div>
        """, unsafe_allow_html=True)

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
        <p class="author-title">Criado por <a href="https://raguiar.eng.br" target="_blank" rel="noopener noreferrer">Rodrigo Aguiar</a></p>
        <div class="sidebar-social-links">
            <a href="https://github.com/RAguiarEng" target="_blank" rel="noopener noreferrer" title="GitHub">
                <i class="fa-brands fa-github"></i>
            </a>
            <a href="https://www.linkedin.com/in/rsouzaaguiar" target="_blank" rel="noopener noreferrer" title="LinkedIn">
                <i class="fa-brands fa-linkedin"></i>
            </a>
        </div>
        <p class="location-text">© 2026 Rodrigo Souza Aguiar · Jaraguá do Sul, SC, Brasil (GMT-3)</p>
    </div>
    """, unsafe_allow_html=True)



logger.info("Aplicativo Streamlit iniciado.")
