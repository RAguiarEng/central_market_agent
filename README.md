# Agente de Inteligência Artificial

Entregável do **Programa ONE** para certificação.

![Programa ONE](img/programa_ONE.png)

Em parceria com a **[Alura](https://www.alura.com.br/)**, O **[Programa ONE](https://www.oracle.com/br/education/oracle-next-education/) - Oracle Next Education - ** proporcionou:

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
├── config.py           # Configurações centrais (paths, modelos, chunking)
├── indexing.py         # Carregamento de PDFs e criação/cache do índice FAISS
├── rag_strategies.py   # Implementação das chains de RAG
├── rag.py              # Entry point — executa e compara as estratégias
├── rag_test.py         # Script de prototipagem (script monolítico original)
├── requirements.txt    # Dependências fixadas com versões exatas
├── .env                # Variáveis de ambiente (NÃO comitar)
├── .gitignore          # Ignora .env e faiss_index/
└── docs/
    └── Mecado_Central_24h/
        ├── FAQ_Clientes_Funcionarios.pdf
        ├── Manual_Fornecedores_Politica_Compras.pdf
        ├── Politica_Atendimento_Trocas_Devolucoes.pdf
        └── Regulamento_Interno_Procedimentos_Operacionais.pdf
```

> O índice FAISS (`faiss_index/`) é gerado localmente e ignorado pelo `.gitignore`.

---

## 🏗️ Arquitetura

Três tipos de RAG são avaliados para encontrar a melhor arquitetura de atendimento do agente.

### RAG Simples

**Futuramente, colocar uma imagem do LangSmith.**

### RAG com Reescrita de Query

**Futuramente, colocar uma imagem do LangSmith.**

> A query reescrita é usada **somente** na busca. O prompt final sempre recebe a pergunta original do usuário.

### RAG Multi-Query *(implementado em `rag_strategies.py`)*

**Futuramente, colocar uma imagem do LangSmith.**

Gera 3 variações da pergunta original → busca em paralelo → deduplica documentos → responde com a query original.


| Estratégia | Descrição |
|---|---|
| **RAG Simples** | A query original do usuário é usada diretamente na busca vetorial |
| **RAG com Reescrita** | Um LLM leve reescreve a query antes da busca, otimizando a recuperação semântica |
| **RAG Multi-Query** | Gera múltiplas variações da pergunta, amplia a cobertura e deduplica os documentos recuperados |

Todas as execuções são rastreadas via **LangSmith**, permitindo comparar os traces lado a lado.


---

## ⚙️ Pré-requisitos

- Python 3.11+
- [Ollama](https://ollama.com/) instalado e rodando localmente
- Modelos Ollama baixados:
  ```bash
  ollama pull llama3.2
  ollama pull bge-m3
  ollama pull gemma3:1b
  ```
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

# 6. Execute
python rag.py
```

---

## 🔑 Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
HF_TOKEN=seu_token_huggingface

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=sua_chave_langsmith
LANGSMITH_PROJECT="RAG_Oracle"
```

> ⚠️ **Nunca comite o arquivo `.env`!** Ele já está no `.gitignore`.

---

## 🧩 Modelos Utilizados

| Papel | Modelo | Onde roda |
|---|---|---|
| Embeddings | `bge-m3` (BAAI) | Ollama (local) |
| LLM principal (resposta) | `llama3.2` | Ollama (local) |
| LLM auxiliar (reescrita) | `gemma3:1b` | Ollama (local) |
| Tokenizer (chunking) | `BAAI/bge-m3` | HuggingFace Hub |

---

## 📊 Rastreamento com LangSmith

Cada execução do `rag.py` envia dois traces para o LangSmith:
- `rag_simples` — com tags `["rag_simples", "sem_reescrita"]`
- `rag_com_reescrita` — com tags `["rag_com_reescrita", "reescrita"]`

Isso permite comparar, lado a lado, a qualidade dos documentos recuperados e das respostas geradas por cada abordagem.

---

## 📄 Base de Conhecimento

Os documentos utilizados pertencem a uma empresa fictícia chamada **Mercado Central 24h** e cobrem:

- FAQ de Clientes e Funcionários
- Manual de Fornecedores e Política de Compras
- Política de Atendimento, Trocas e Devoluções
- Regulamento Interno e Procedimentos Operacionais

---

## 📝 Licença

Projeto de estudo — sem licença formal. Uso livre para fins educacionais.