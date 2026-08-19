[![Português](https://img.shields.io/badge/Idioma-Portugu%C3%AAs-blue.svg)](README.md)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-1C3C3C?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-005571?style=flat)
![Cohere](https://img.shields.io/badge/Cohere-Embeddings-39594C?style=flat)
![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM%20Gateway-000000?style=flat)
![Oracle Cloud Infrastructure](https://img.shields.io/badge/Oracle%20Cloud-Infrastructure-F80000?style=flat&logo=oracle&logoColor=white)
![Nginx](https://img.shields.io/badge/Nginx-Reverse%20Proxy-009639?style=flat&logo=nginx&logoColor=white)
![Let's Encrypt](https://img.shields.io/badge/HTTPS-Let's%20Encrypt-003A70?style=flat&logo=letsencrypt&logoColor=white)
![systemd](https://img.shields.io/badge/systemd-Service%20Manager-white?style=flat&logo=linux&logoColor=black)

<h1 align="center">Supervisor-Specialists Agent: RAG Applied to 24h Central Market Customer Service</h1>

<h3 align="center">Deliverable of the ONE Program - G10</h3>

<p align="center">
  <img src="img/programa_ONE.png" alt="ONE Program" height="400">
</p>

In partnership with <a href="https://www.alura.com.br/" target="_blank" rel="noopener noreferrer">Alura</a>, the <a href="https://www.oracle.com/br/education/oracle-next-education/" target="_blank" rel="noopener noreferrer">ONE Program</a> (Oracle Next Education) provided:

- **Foundations:** GitHub, Applied Python, AI Fundamentals, and Machine Learning.
- **Autonomous Agents and Automation with n8n**
- **AI Engineering and RAG with LangChain**
- **Oracle Cloud Infrastructure (OCI):** Cloud deployment of the project.

---

## 🎯 Objective

Build an **AI Agent** focused on **answering employee questions at a 24H Central Market** regarding **various pertinent documents** within the company's domain.

The Agent is accessible to all company employees without requiring restricted access.

--- 

## 🌐 Live Application

Access: [app.rsa.ia.br](https://app.rsa.ia.br)

---

## 🗂️ Project Structure

```
RAG/
├── agents/
│   ├── specialist/
│   │   ├── __init__.py
│   │   └── base.py                       # Base class for specialist agents
│   └── supervisor/
│       ├── __init__.py
│       ├── agent.py                      # Supervisor Agent (router)
│       ├── models.py                     # Pydantic schemas (AgentSelection)
│       └── specialist_summaries.json     # Metadata and scopes for each specialist
├── core/
│   ├── __init__.py
│   ├── protocols.py                      # Agent Communication Protocol (ACPMessage)
│   └── state.py                          # Global state structure (AgentState)
├── docs/
│   └── Mercado_Central_24h/              # Knowledge base PDF documents
│       ├── FAQ_Clientes_Funcionarios.pdf
│       ├── Manual_Fornecedores_Politica_Compras.pdf
│       ├── Politica_Atendimento_Trocas_Devolucoes.pdf
│       └── Regulamento_Interno_Procedimentos_Operacionais.pdf
├── faiss_index/                          # Automatically generated FAISS vector indices
├── app.py                                # Streamlit graphical interface (Entry Point)
├── config.py                             # Global configurations, paths, and models
├── indexing.py                           # FAISS index creation and loading
├── rag_multiagent.py                     # LangGraph workflow graph definition
├── requirements.txt                      # Project dependencies
├── .env                                  # API keys and environment variables
├── .env.example                          # Example .env configuration file
└── .gitignore                            # Files ignored by version control
```

> The FAISS index (`faiss_index/`) is generated locally and ignored by `.gitignore`.

---

## 🏗️ Architecture

The current architecture is based on a **Supervisor-Specialists** pattern managed by **LangGraph**:

1. **Supervisor Agent (`SupervisorAgent`):**

- Analyzes the user's question and decides which specialist to route the request to, based on structured scope summaries defined in `specialist_summaries.json`.
- If the question does not match any specialty, it routes the query to a general handler (`general`).
- Implemented using the [`nvidia/nemotron-3.5-lightning:free`](https://openrouter.ai/nvidia/nemotron-3.5-lightning:free) model via OpenRouter.

2. **Specialist Agents (`SpecialistAgent`):**

- Each specialist manages a vector index (FAISS) created from a specific PDF document.
- They perform local semantic searches (RAG) using the [`embed-multilingual-v3.0`](https://docs.cohere.com/docs/cohere-embed) model via OpenRouter and generate a targeted response.
- Implemented using the [`nvidia/nemotron-3-super-120b-a12b:free`](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free) model via OpenRouter.

3. **Streamlit Interface (`app.py`):**

- Interactive chat displaying the conversation flow and real-time metrics (latency, token consumption, and estimated cost).

---

## ⚙️ Prerequisites

- Python 3.11+
- `API Key` for OpenRouter and Cohere
- [LangSmith](https://smith.langchain.com/) account (optional, but recommended for tracing)

---

## 🚀 Installation and Usage

```bash
# 1. Clone the repository
git clone <repository-url>
cd RAG

# 2. Create and activate the virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
# Copy .env.example and populate your keys in .env
cp .env.example .env

# 5. Place your PDF files in the docs/ directory
# (the subfolder structure is automatically loaded by DirectoryLoader)
```

> ⚠️ **Never commit your `.env` file!** It is already listed in `.gitignore`.

---

## 🧩 Models Used

| Role | Model | Host / Provider |
|---|---|---|
| Chunking and Embeddings | `embed-multilingual-v3.0` | [Cohere](https://docs.cohere.com/docs/cohere-embed) |
| Primary LLM (Responses) | `nvidia/nemotron-3-super-120b-a12b:free` | [OpenRouter](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free) |
| Supervisor LLM (Routing) | `nvidia/nemotron-3.5-lightning:free` | [OpenRouter](https://openrouter.ai/nvidia/nemotron-3.5-lightning:free) |

---

## 📊 Tracing with LangSmith

The following images illustrate how LangSmith logs the execution chain.

`Input/User`: The question submitted by the user.

`Output/AI`: The returned response.

![agente_supervisor](img/langsmith_supervisor.png)

This image shows the token consumption of the supervisor agent (`supervisor_router`) for both *input* and *output*, along with execution times for each step in its chain and the model used.

![agente_especialista](img/langsmith_especialista.png)

In this trace of the specialist agent (`manual_fornecedores_compras`), the information retrieval step (`VectorStoreRetriever`) is also visible. The supervisor agent invoked it based on the context of the incoming question.

The following images showcase the complete execution flow of the chain.

**Example 01:**

![langsmith_01](img/langsmith_01.png)

**Example 02:**

![langsmith_02](img/langsmith_02.png)

LangSmith tracing provides a significant advantage for evaluating the project, clearly understanding the execution path, and tracking associated resource consumption (latency and tokens).

---

## 📊 Metrics Dashboard

The parameters gathered through LangSmith can be presented as user-facing metrics. This brings full transparency to the costs of each request and helps identify which agent was responsible for delivering the response.

![painel_metricas](img/painel_metricas.png)

---

## 📄 Knowledge Base

The documents used belong to a fictional company named **Mercado Central 24h** and cover:

- Customer and Employee FAQ
- Supplier Manual and Purchasing Policy
- Customer Service, Exchange, and Return Policy
- Internal Regulations and Operational Procedures

The `specialist_summaries.json` file was created from the metadata and summaries of these documents to serve as a routing reference for the specialist agent. I adopted this strategy to avoid lengthy explanatory prompts, thereby reducing latency and token costs during task routing. Furthermore, the metadata contained in this `.json` file also enables curation of the reference `.pdf` files, as shown in these screenshots:

**Example 01:**

![chat_versao_arquivos](img/chat_versao_arquivos.png)

**Example 02:**

![chat_depto_arquivos](img/chat_depto_arquivos.png)

This strategy is also useful for allowing the supervisor agent to handle generic questions—those not found within the document contents. If an answer cannot be found even in the `.json` file, the supervisor returns an apology message or guides the user to seek further information from other sources, such as the market's website.

---

## 🌐 Production Deployment (Oracle Cloud Infrastructure)

The application is deployed in production at **[https://app.rsa.ia.br](https://app.rsa.ia.br)**, hosted on a compute instance (VM) on **Oracle Cloud Infrastructure (OCI)**.

### Infrastructure Architecture

```
User → DNS (app.rsa.ia.br) → OCI (Security List + NSG) → Ubuntu Instance

├── iptables (OS firewall) 
├── Nginx (reverse proxy + SSL) 
└── Streamlit (managed via systemd, port 8501)
```

- **DNS:** The `app.rsa.ia.br` subdomain points to the OCI instance's public IP, while the apex domain `rsa.ia.br` remains hosted separately on GitHub Pages.
- **OCI Networking:** HTTP (80) and HTTPS (443) traffic allowed in the VCN *Security List* and the *Network Security Group (NSG)* associated with the instance.
- **Operating System Firewall (`iptables`):** `ACCEPT` rules added for ports 80, 443, and 8501, persisted with `netfilter-persistent`.
- **Nginx:** Acts as a *reverse proxy*, forwarding incoming requests on `app.rsa.ia.br` to the Streamlit application running locally on port `8501`.
- **HTTPS:** Free SSL certificate issued via **Let's Encrypt (Certbot)**, with scheduled automatic renewal (`certbot.timer`).
- **Streamlit Persistence:** The application is managed by a **`systemd`** service, ensuring it starts automatically on system boot and restarts automatically in case of failure.

### Nginx Configuration

- File: `/etc/nginx/sites-available/streamlit_app`

```bash
server {
    listen 80;
    listen [::]:80;
    server_name app.rsa.ia.br;

    location / {
        proxy_pass http://localhost:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

> After running Certbot (`sudo certbot --nginx -d app.rsa.ia.br`), this file is automatically updated with the `listen 443 ssl;` blocks and the HTTP-to-HTTPS redirect.

### Streamlit `systemd` Service

I configured a `systemd` service so that Streamlit starts automatically whenever the instance is rebooted, and restarts itself upon failure.

- File: `/etc/systemd/system/streamlit.service`

```bash
[Unit]
Description=Streamlit App - Central Market Agent
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/central_market_agent
ExecStart=/home/ubuntu/central_market_agent/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

After creating the file, I enabled the service to start with the operating system and launched it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit.service
sudo systemctl start streamlit.service
```

**Useful maintenance commands:**

```bash
# Check service status
sudo systemctl status streamlit.service

# Manually restart service
sudo systemctl restart streamlit.service

# View real-time logs
sudo journalctl -u streamlit.service -f

# Test SSL certificate renewal (dry run)
sudo certbot renew --dry-run
```

---

## 🔄 Updating the OCI Instance after GitHub Changes

Whenever I push new changes to the GitHub repository, I need to repeat the process below on the OCI instance so that the production application reflects the latest codebase.

**Step 1: Connect via SSH to the instance**

```bash
ssh ubuntu@<instance_IP>
```

**Step 2: Navigate to the project folder**

```bash
cd ~/central_market_agent
```

**Step 3: Fetch updates from the repository**

```bash
git pull origin main
```

> Replace `main` with the target branch name if using a different branch.

**Step 4: Activate the virtual environment**

```bash
source .venv/bin/activate
```

**Step 5: Update dependencies (if `requirements.txt` has changed)**

```bash
pip install -r requirements.txt
```

**Step 6: Restart the Streamlit service**

Since the application is managed by `systemd`, there is no need to kill processes manually — simply restart the service:

```bash
sudo systemctl restart streamlit.service
```

**Step 7: Verify that the service is running properly**

```bash
sudo systemctl status streamlit.service
```

**Step 8: Test the application**

Open `https://app.rsa.ia.br` in your browser and confirm that changes have been applied correctly.

### Quick Summary (sequential commands)

```bash
cd ~/central_market_agent
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart streamlit.service
sudo systemctl status streamlit.service
```

---

## 🧩 Challenges

During the deployment of the application to production on OCI, I encountered a series of roadblocks across different network and infrastructure layers. Below is the documentation of the diagnosis and resolution for each, in the order they were identified:

| # | Challenge | Diagnosis | Solution |
|---|---|---|---|
| 1 | External connection timing out | Missing route rule associated with the *Internet Gateway* in the VCN *Route Table* | Added the route `0.0.0.0/0 → Internet Gateway` |
| 2 | External connection still timing out | Instance *Network Security Group (NSG)* lacked ingress rules for ports 80/443 | Added ingress rules allowing ports 80 and 443 in the NSG |
| 3 | Local connection working, but external blocked | OS `iptables` only allowed port 22 (SSH) for new connections | Added `ACCEPT` rules for ports 80, 443, and 8501 in `iptables`, persisted with `iptables-persistent` |
| 4 | Nginx displaying default page instead of Streamlit | Default Nginx site still active, conflicting with domain configuration | Removed the symlink `/etc/nginx/sites-enabled/default` |
| 5 | Nginx returning 404 on domain root | `location` block configured for `/streamlit-app/` instead of root `/` | Adjusted configuration to `location /`, aligned with dedicated subdomain strategy |
| 6 | `502 Bad Gateway` error | Streamlit process was not running (had been terminated alongside previous SSH session) | Restarted process, then migrated to a `systemd` service for a permanent fix |
| 7 | `streamlit` command not found (`Exit 127`) | Virtual environment (`.venv`) was not active in terminal session | Activated `.venv` prior to running, and referenced the absolute binary path in `systemd` service |

After resolving this entire sequence of root causes, the application was successfully published over HTTPS, featuring automatic certificate renewal and automatic recovery upon failures or instance reboots.

---

## 🚧 Future Improvements

This project is under continuous development. Key areas I intend to address in future iterations:

- **Latency reduction in fallback flow:** When the Supervisor Agent does not identify a suitable specialist for the incoming question, the response time of the general flow (`general`) remains elevated.
- **Evolution of the project test harness:** Improve the organization of testing, evaluation, and validation for agent behavior.
- **LLM usage and rate limiting:** As the application is publicly accessible, implement rate-limiting mechanisms to preserve the sustainability and quota of API keys.
- **Support for multiple API keys and per-user model selection:** Evaluate ways to allow different users to use their own credentials and/or preferred LLM models, increasing flexibility and scalability.

---

## 📝 License

Study project — under the MIT License. Free for educational purposes.
