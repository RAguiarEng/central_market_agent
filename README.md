<h1 align="center">Agente de Inteligência Artificial</h1>

<h3 align="center">Entregável do Programa ONE para certificação.</h3>

<p align="center">
  <img src="img/programa_ONE.png" alt="Programa ONE" height="400">
</p>

Em parceria com a <a href="https://www.alura.com.br/" target="_blank" rel="noopener noreferrer">Alura</a>, o <a href="https://www.oracle.com/br/education/oracle-next-education/" target="_blank" rel="noopener noreferrer">Programa ONE</a> proporcionou:

- **Nivelamento:** GitHub, Python aplicado, Fundamentos de IA e Machine Learning.
- **Agentes Autônomos e Automação com n8n**
- **Engenharia de IA e RAG com LangChain**
- **Oracle Cloud Infrasctructure (OCI):** disponibilização do projeto na nuvem.  

---

## 🎯 Objetivo

Construir um **Agente de IA** focado em **responder perguntas de colaboradores de um Mercado Central 24H** em relação a **diversos documentos** pertinentes no contexto da empresa.

O Agente está aberto a qualquer colaborador da empresa, sem necessidade de acesso restrito.

---

## 🗂️ Estrutura do Projeto

```
RAG/
├── agents/
│   ├── specialist/
│   │   ├── __init__.py
│   │   └── base.py                       # Classe base dos agentes especialistas
│   └── supervisor/
│       ├── __init__.py
│       ├── agent.py                      # Agente Supervisor (roteador)
│       ├── models.py                     # Estruturas Pydantic (AgentSelection)
│       └── specialist_summaries.json     # Metadados e escopos de cada especialista
├── core/
│   ├── __init__.py
│   ├── protocols.py                      # Protocolo de Comunicação de Agentes (ACPMessage)
│   └── state.py                          # Estrutura do estado global (AgentState)
├── docs/
│   └── Mercado_Central_24h/              # Documentos PDF de base de conhecimento
│       ├── FAQ_Clientes_Funcionarios.pdf
│       ├── Manual_Fornecedores_Politica_Compras.pdf
│       ├── Politica_Atendimento_Trocas_Devolucoes.pdf
│       └── Regulamento_Interno_Procedimentos_Operacionais.pdf
├── faiss_index/                          # Índices FAISS gerados automaticamente
├── app.py                                # Interface gráfica em Streamlit (Entry Point)
├── config.py                             # Configurações globais, paths e modelos
├── indexing.py                           # Criação e carregamento dos índices FAISS
├── rag_multiagent.py                     # Definição do grafo de fluxo LangGraph
├── requirements.txt                      # Dependências do projeto
├── .env                                  # Chaves de API e configurações de ambiente
└── .gitignore                            # Arquivos ignorados pelo controle de versão
```

> O índice FAISS (`faiss_index/`) é gerado localmente e ignorado pelo `.gitignore`.

---

## 🏗️ Arquitetura

A arquitetura atual é baseada em um padrão **Supervisor-Especialistas** gerenciado pelo **LangGraph**:

1. **Agente Supervisor (`SupervisorAgent`):**

- Analisa a pergunta do usuário e decide para qual especialista direcionar a solicitação com base em resumos de escopo estruturados em `specialist_summaries.json`.
- Se a pergunta não se encaixar em nenhuma especialidade, ele direciona para um manipulador geral (`general`).
- Implementado usando o modelo [`nvidia/nemotron-3.5-lightning:free`](https://openrouter.ai/nvidia/nemotron-3.5-lightning:free) via OpenRouter.

2. **Agentes Especialistas (`SpecialistAgent`):**

- Cada especialista gerencia um índice vetorial (FAISS) criado a partir de um documento PDF específico.
- Eles realizam buscas semânticas locais (RAG) utilizando o modelo [`embed-multilingual-v3.0`](https://docs.cohere.com/docs/cohere-embed), via OpenRouter, e geram uma resposta focada.
- Implementados usando o modelo [`nvidia/nemotron-3-super-120b-a12b:free`](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free) via OpenRouter.

3. **Interface Streamlit (`app.py`):**

- Chat interativo que exibe o andamento da conversa e apresenta métricas em tempo real (latência, consumo de tokens e custo estimado).


---

## ⚙️ Pré-requisitos

- Python 3.11+
- `API Key` da OpenRouter e do Cohere
- Conta no [LangSmith](https://smith.langchain.com/) (opcional, mas recomendado para rastreamento)

---

## 🚀 Instalação e Uso

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd RAG

# 2. Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
# Copie o arquivo de exemplo e preencha com suas chaves:
cp .env.example .env

# 5. Coloque seus PDFs na pasta docs/
# (a estrutura de subpastas é carregada automaticamente pelo DirectoryLoader)
```

---

## 🔑 Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
HF_TOKEN=seu_token_huggingface

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=sua_chave_langsmith
LANGSMITH_PROJECT="Nome_do_seu_projeto"
```

> ⚠️ **Nunca comite o arquivo `.env`!** Ele já está listado no `.gitignore`.

---

## 🧩 Modelos Utilizados

| Papel | Modelo | Onde roda |
|---|---|---|
| Chunking e Embeddings | `embed-multilingual-v3.0` | [Cohere](https://docs.cohere.com/docs/cohere-embed) |
| LLM principal (resposta) | `nvidia/nemotron-3-super-120b-a12b:free` | [OpenRouter](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free) |
| LLM supervisor (roteamento) | `nvidia/nemotron-3.5-lightning:free` | [OpenRouter](https://openrouter.ai/nvidia/nemotron-3.5-lightning:free) |
---

## 📊 Rastreamento com LangSmith

### Registro da cadeia em execução

#### Exemplo 01

![langsmith_01](img/langsmith_01.png)

#### Exemplo 02

![langsmith_02](img/langsmith_02.png)

---

## 📄 Base de Conhecimento

Os documentos utilizados pertencem a uma empresa fictícia chamada **Mercado Central 24h** e cobrem:

- FAQ de Clientes e Funcionários
- Manual de Fornecedores e Política de Compras
- Política de Atendimento, Trocas e Devoluções
- Regulamento Interno e Procedimentos Operacionais

O arquivo `agents\supervisor\specialist_summaries.json` foi criado a partir dos metadados e dos sumários desses arquivos para servir como referência de roteamento para o agente especialista.

---

## 📝 Licença

Projeto de estudo — com licença MIT. Uso livre para fins educacionais.