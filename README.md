# 🤖 GenAI Master Samples

> **Projeto educacional completo** para aprender a criar **Agentes de IA** com LangChain, FastAPI e Streamlit.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Índice

- [🎯 Sobre o Projeto](#-sobre-o-projeto)
- [🗺️ Trilha de Aprendizado](#️-trilha-de-aprendizado)
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📁 Estrutura do Projeto](#-estrutura-do-projeto)
- [🤖 Agentes Disponíveis](#-agentes-disponíveis)
- [🧠 Skills (Habilidades)](#-skills-habilidades)
- [🔧 Tools (Ferramentas)](#-tools-ferramentas)
- [🌐 API REST](#-api-rest)
- [🎮 Demo Interativo](#-demo-interativo)
- [📚 Conceitos Importantes](#-conceitos-importantes)
- [🛠️ Criando Seus Próprios Componentes](#️-criando-seus-próprios-componentes)
- [🐳 Docker](#-docker)
- [🔑 Configuração](#-configuração)
- [📖 Exemplos de Uso](#-exemplos-de-uso)
- [🤝 Contribuindo](#-contribuindo)

---

## 🎯 Sobre o Projeto

Este projeto foi desenvolvido para ensinar os conceitos fundamentais de **Agentes de IA**:

| Conceito | O que você vai aprender |
|----------|------------------------|
| 🤖 **Agentes** | Programas que usam LLMs para "pensar" e agir autonomamente |
| 🔧 **Tools** | Como permitir que o agente execute ações reais (cálculos, buscas, APIs) |
| 🧠 **Skills** | Habilidades de alto nível que compõem múltiplas tools (pesquisa, resumo, criação) |
| 📚 **RAG** | Como dar conhecimento específico ao agente com documentos |
| 🧠 **Memória** | Como manter contexto entre conversas (curto e longo prazo) |
| 🔌 **MCP** | Model Context Protocol para conectar a servidores externos |
| 🌐 **API** | Como expor agentes via REST API com streaming |

---

## 🗺️ Trilha de Aprendizado

Siga esta trilha para aprender **do zero ao avançado** sobre Agentes de IA com este projeto.
Cada etapa constrói sobre a anterior — ao final, você terá domínio completo!

### 🟢 Nível 1 — Fundamentos (Primeiros Passos)

> _"Entender o básico: conversar com um LLM e configurar o ambiente."_

| Etapa | O que fazer | Arquivo(s) | Conceito |
|:-----:|-------------|------------|----------|
| 1.1 | Instalar o projeto e configurar o `.env` | `README.md`, `.env.example` | Setup do ambiente |
| 1.2 | Subir o Streamlit e conversar com o **Simple Agent** | `app.py`, `agents/simple_agent.py` | Chat básico com LLM |
| 1.3 | Ler e entender a **classe base abstrata** | `agents/base_agent.py` | Herança, ABC, Template Pattern |
| 1.4 | Trocar entre **OpenAI ↔ Gemini** na sidebar | `agents/simple_agent.py` | Multi-provider, API keys |
| 1.5 | Explorar os **templates de prompts** | `templates/prompts.py` | System Prompt, Guardrails |

**✅ Ao final:** Você sabe conversar com um LLM, trocar de provedor e personalizar o comportamento do agente.

---

### 🟡 Nível 2 — Tools (Ferramentas)

> _"Dar superpoderes ao agente: cálculos, buscas, APIs externas."_

| Etapa | O que fazer | Arquivo(s) | Conceito |
|:-----:|-------------|------------|----------|
| 2.1 | Usar o **OpenAI Agent** e perguntar "Quanto é 15% de 230?" | `agents/openai_agent.py` | ReAct Pattern, Tool Calling |
| 2.2 | Estudar como uma **tool é criada** | `tools/calculator.py` | `@tool` decorator, Pydantic schema |
| 2.3 | Testar a **busca web** perguntando sobre notícias | `tools/web_search.py` | DuckDuckGo, busca gratuita |
| 2.4 | Usar o **Gemini Agent** com as mesmas tools | `agents/gemini_agent.py` | Mesmo padrão, provider diferente |
| 2.5 | Explorar **tools de finanças** (crypto, ações, câmbio) | `tools/crypto.py`, `tools/stocks.py` | APIs externas (CoinGecko, Alpha Vantage) |
| 2.6 | Criar sua **própria tool** seguindo o padrão | `tools/` | Extensibilidade do sistema |

**✅ Ao final:** Você sabe como Tools funcionam, como o agente decide usá-las (ReAct) e como criar as suas.

---

### 🟠 Nível 3 — RAG (Base de Conhecimento)

> _"Dar conhecimento específico ao agente com seus próprios documentos."_

| Etapa | O que fazer | Arquivo(s) | Conceito |
|:-----:|-------------|------------|----------|
| 3.1 | Fazer upload de um **PDF** na sidebar do Streamlit | `app.py` (seção RAG) | Upload + processamento |
| 3.2 | Entender como documentos são **divididos em chunks** | `knowledge_base/document_loader.py` | Chunking, overlap |
| 3.3 | Estudar como **embeddings** transformam texto em vetores | `knowledge_base/vector_store.py` | Embeddings, FAISS |
| 3.4 | Testar com **diferentes provedores de embeddings** | `.env` → `EMBEDDING_PROVIDER` | OpenAI, Gemini, Ollama, HuggingFace |
| 3.5 | Perguntar sobre o conteúdo do documento ao agente | `tools/rag_tool.py` | Similarity search, contexto |
| 3.6 | Salvar e **carregar a base do disco** | Sidebar → Armazenamento | Persistência de vector store |

**✅ Ao final:** Você sabe fazer RAG completo — upload, chunking, embeddings, busca semântica e integração com o agente.

---

### 🔴 Nível 4 — Agentes Especialistas

> _"Criar agentes focados em domínios específicos."_

| Etapa | O que fazer | Arquivo(s) | Conceito |
|:-----:|-------------|------------|----------|
| 4.1 | Usar o **Finance Agent** e consultar crypto/ações | `agents/finance_agent.py` | Agente de domínio |
| 4.2 | Usar o **Knowledge Agent** para buscar na Wikipedia | `agents/knowledge_agent.py` | Tools especializadas |
| 4.3 | Usar o **Web Search Agent** para pesquisas | `agents/websearch_agent.py` | Single-tool agent |
| 4.4 | Comparar **system prompts** de cada especialista | `templates/prompts.py` | Prompt engineering por domínio |
| 4.5 | Experimentar **RAG + Especialista** juntos | Sidebar → Upload + Agente | Combinação de capacidades |
| 4.6 | Entender o conceito de **Skills vs Tools** | `skills/base_skill.py`, `skills/__init__.py` | Abstração de alto nível |
| 4.7 | Usar o **Skills Agent** (pesquisa, resumo, criação) | `agents/skills_agent.py` | Skills compostas |
| 4.8 | Estudar como uma **Skill compõe múltiplas Tools** | `skills/research_skill.py` | Composição, multi-etapa |
| 4.9 | Criar sua **própria Skill** seguindo o padrão | `skills/` | Extensibilidade avançada |

**✅ Ao final:** Você sabe criar agentes focados com tools, prompts e Skills especializadas.

---

### 🟣 Nível 5 — Memória e Contexto

> _"Fazer o agente lembrar de conversas e informações importantes."_

| Etapa | O que fazer | Arquivo(s) | Conceito |
|:-----:|-------------|------------|----------|
| 5.1 | Ativar **memória de curto prazo** e conversar | Sidebar → Memória | Últimas N mensagens |
| 5.2 | Ativar **memória de longo prazo** e testar persistência | `core/memory.py` | Armazenamento em disco |
| 5.3 | Usar **memória combinada** (curto + longo) | Sidebar → Combinada | Estratégia híbrida |
| 5.4 | Reiniciar o app e verificar que memória **persiste** | Terminal → restart | Persistência entre sessões |
| 5.5 | Estudar a implementação de cada tipo | `core/memory.py` | Classes de memória |

**✅ Ao final:** Você sabe como funcionam os 3 tipos de memória e quando usar cada um.

---

### ⚫ Nível 6 — Infraestrutura (Local e Cloud)

> _"Rodar modelos localmente, expor via API e containerizar."_

| Etapa | O que fazer | Arquivo(s) | Conceito |
|:-----:|-------------|------------|----------|
| 6.1 | Instalar **Ollama** e rodar modelos **100% locais** | `agents/ollama_agent.py` | Privacidade, sem API key |
| 6.2 | Usar o **Azure OpenAI Agent** (compliance empresarial) | `agents/azure_agent.py` | SLA, rede privada |
| 6.3 | Subir a **API REST** com FastAPI | `api.py` | Endpoints, sessões, streaming |
| 6.4 | Testar com o **Demo interativo** SSE | `static/chat_sse_demo.html` | Server-Sent Events |
| 6.5 | **Containerizar** com Docker Compose | `Dockerfile`, `docker-compose.yml` | API + Chat UI em containers |
| 6.6 | Explorar a configuração **Kubernetes** | `k8s/` | Deploy em produção |

**✅ Ao final:** Você sabe rodar local, expor via API, fazer streaming e deployar com Docker/K8s.

---

### 🌟 Nível 7 — MCP e Avançado

> _"Conectar a servidores externos e dominar o ecossistema completo."_

| Etapa | O que fazer | Arquivo(s) | Conceito |
|:-----:|-------------|------------|----------|
| 7.1 | Entender o **MCP Demo Agent** (sem conexão real) | `agents/mcp_agent.py` → `MCPAgentDemo` | Conceito do protocolo |
| 7.2 | Usar **MCP real** com Fetch (buscar URLs) | `agents/mcp_agent.py` → `MCPAgent` | Conexão real a servidor MCP |
| 7.3 | Experimentar **MCP Filesystem** (ler/escrever arquivos) | MCP Server: filesystem | Acesso a recursos locais |
| 7.4 | Testar **MCP Time**, **SQLite** e outros | MCP Servers variados | Diversidade de tools externas |
| 7.5 | Estudar o **registro dinâmico** de agentes na API | `api.py` → `AgentRegistry` | Descoberta automática |
| 7.6 | Combinar **tudo**: agente + tools + RAG + memória + MCP | Projeto completo | Arquitetura completa |

**✅ Ao final:** Você domina o ecossistema completo de Agentes de IA — do zero ao avançado! 🎓

---

### 📊 Mapa Visual da Trilha

```
🟢 Nível 1          🟡 Nível 2          🟠 Nível 3
Fundamentos    →    Tools          →    RAG
• Setup             • ReAct Pattern     • Embeddings
• Simple Agent      • @tool decorator   • Vector Store
• Providers         • APIs externas     • Chunking
     │                   │                   │
     ▼                   ▼                   ▼
🔴 Nível 4          🟣 Nível 5          ⚫ Nível 6
Especialistas  →    Memória        →    Infraestrutura
• Finance Agent     • Curto prazo       • Ollama (local)
• Skills Agent      • Longo prazo       • Azure OpenAI
• Skills vs Tools   • Persistência      • API + Docker
     │                   │                   │
     └───────────────────┼───────────────────┘
                         ▼
                   🌟 Nível 7
                   MCP & Avançado
                   • Servidores externos
                   • Arquitetura completa
                   • 🎓 Master GenAI!
```

---

## ✨ Features

### 🖥️ Interfaces
- ✅ **Streamlit App** - Interface completa estilo ChatGPT
- ✅ **API REST** - FastAPI com documentação automática
- ✅ **Demo Web** - Chat interativo com SSE streaming
- ✅ **3 Temas** - Default, ChatGPT e Gemini

### 🤖 Agentes
- ✅ **OpenAI** - GPT-4, GPT-4o, GPT-4o-mini
- ✅ **Google Gemini** - Gemini 2.5 Flash, 2.0 Flash, 1.5 Pro
- ✅ **Azure OpenAI** - Mesmos modelos GPT com compliance empresarial e SLA
- ✅ **Ollama (Local)** - Llama 3.2, Mistral, CodeLlama, Phi-3, Gemma, etc. **(sem API key!)**
- ✅ **Especializados** - Finance, Knowledge, Web Search
- ✅ **Skills Agent** - Habilidades avançadas: Pesquisa, Resumo e Criação de Conteúdo (Azure OpenAI)
- ✅ **MCP** - Fetch, Filesystem, Memory, Time, SQLite, Brave Search, GitHub

### 🧠 Skills (Habilidades)
- ✅ **Research Skill** - Pesquisa aprofundada (Web + Wikipedia + síntese)
- ✅ **Summarize Skill** - Análise e resumo inteligente (executivo, pontos-chave, crítico)
- ✅ **Content Creation Skill** - E-mails, relatórios e posts (LinkedIn, Twitter/X)

### 🔧 Tools
- ✅ Calculadora, Data/Hora, Busca Web
- ✅ Geocoding, Criptomoedas, Ações/Forex
- ✅ Wikipedia, RAG Search

### 📚 RAG
- ✅ Upload de PDF, DOCX, CSV, TXT, MD, JSON
- ✅ Vector Store com FAISS
- ✅ Chunking configurável

---

## 🚀 Quick Start

### Pré-requisitos

- Python 3.11+
- Poetry (recomendado) ou pip
- API Key da OpenAI e/ou Google

### 1️⃣ Clone o repositório

```bash
git clone https://github.com/seu-usuario/GenAI_Master_Samples.git
cd GenAI_Master_Samples
```

### 🐍 Criando um Ambiente Virtual (.venv)

É altamente recomendado usar um **ambiente virtual** para isolar as dependências do projeto:

```bash
# Crie o ambiente virtual na raiz do projeto:
python3 -m venv .venv

# Ative o ambiente virtual:

# macOS / Linux:
source .venv/bin/activate

# Windows (CMD):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

Após ativar, você verá `(.venv)` no início do terminal:

```bash
(.venv) ➜ GenAI_Master_Samples $
```

> 💡 **Dicas**:
> - Sempre ative o `.venv` antes de instalar dependências ou executar o projeto.
> - Para desativar o ambiente virtual, basta digitar: `deactivate`
> - O diretório `.venv/` já está incluído no `.gitignore` — não será versionado.

#### Configurando o Poetry para usar o .venv do projeto

Se estiver usando o **Poetry**, configure-o para criar o ambiente virtual dentro do próprio projeto:

```bash
poetry config virtualenvs.in-project true
```

Assim, ao rodar `poetry install`, ele usará automaticamente o `.venv/` na raiz do projeto.

### 📦 Instalando o Poetry (Recomendado)

O projeto utiliza o **Poetry** para gerenciar dependências. Caso ainda não tenha instalado:

```bash
# macOS / Linux (método oficial):
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell):
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Ou via pipx (alternativa):
pipx install poetry

# Ou via Homebrew (macOS):
brew install poetry
```

Após a instalação, verifique:

```bash
poetry --version
```

> 💡 **Dica**: Caso o comando `poetry` não seja encontrado, adicione ao PATH:
> ```bash
> # macOS / Linux — adicione ao ~/.zshrc ou ~/.bashrc:
> export PATH="$HOME/.local/bin:$PATH"
> ```

### 2️⃣ Instale as dependências

```bash
# Com Poetry (recomendado)
poetry install

# Ou com pip
pip install -r requirements.txt
```

### 3️⃣ Configure as API Keys

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite e adicione suas chaves
nano .env
```

```env
# .env
OPENAI_API_KEY=sk-sua-chave-aqui
GOOGLE_API_KEY=sua-chave-aqui
```

### 🦙 Usando Ollama (Opcional - Gratuito e Local!)

Se preferir rodar modelos **localmente sem API key**:

```bash
# 1. Instale o Ollama (https://ollama.ai)
# macOS:
brew install ollama

# 2. Baixe um modelo (ex: Llama 3.2)
ollama pull llama3.2

# 3. O servidor inicia automaticamente, ou execute:
ollama serve
```

> 💡 **Vantagens do Ollama**: Gratuito, privado (dados não saem do PC), funciona offline!

### 🔨 Instalando o Make (Opcional - para usar o Makefile)

O projeto inclui um **Makefile** com comandos úteis. Para utilizá-lo, instale o `make`:

```bash
# macOS (já vem instalado com Xcode Command Line Tools):
xcode-select --install

# Ou via Homebrew:
brew install make

# Linux (Debian/Ubuntu):
sudo apt update && sudo apt install make

# Linux (Fedora/RHEL):
sudo dnf install make

# Windows (via Chocolatey):
choco install make

# Windows (via Scoop):
scoop install make
```

> 💡 **Dica**: No macOS, o `make` geralmente já está disponível. Teste com `make --version` no terminal.

### 4️⃣ Execute!

```bash
# Usando Makefile (recomendado)
make dev          # Inicia API + Streamlit

# Ou manualmente
make api          # Apenas API (porta 8000)
make app          # Apenas Streamlit (porta 8501)
```

### 5️⃣ Acesse

| Interface | URL |
|-----------|-----|
| 🎮 **Demo Chat** | http://localhost:8000/demo |
| 📚 **API Docs** | http://localhost:8000/docs |
| 🎨 **Streamlit** | http://localhost:8501 |
| 🐳 **Chat UI (Docker)** | http://localhost:8080 |

---

## 📁 Estrutura do Projeto

```
GenAI_Master_Samples/
│
├── 📄 app.py                    # Interface Streamlit
├── 📄 api.py                    # API REST FastAPI
├── 📄 Makefile                  # Comandos úteis
├── 📄 pyproject.toml            # Configuração Poetry
├── 📄 requirements.txt          # Dependências pip
├── 📄 Dockerfile                # Build da API
├── 📄 docker-compose.yml        # Orquestração dos serviços
├── 📄 .env                      # Variáveis de ambiente
│
├── 📁 agents/                   # 🤖 AGENTES DE IA
│   ├── base_agent.py            # Classe base abstrata
│   ├── simple_agent.py          # Agente simples (sem tools)
│   ├── openai_agent.py          # Agente OpenAI completo
│   ├── gemini_agent.py          # Agente Gemini completo
│   ├── azure_agent.py           # ☁️ Agente Azure OpenAI
│   ├── ollama_agent.py          # 🦙 Agente Ollama (local)
│   ├── finance_agent.py         # 💰 Especialista em finanças
│   ├── knowledge_agent.py       # 📚 Especialista em conhecimento
│   ├── websearch_agent.py       # 🔍 Especialista em pesquisa
│   ├── mcp_agent.py             # 🔌 Agente MCP
│   └── skills_agent.py          # 🧠 Agente com Skills (Azure OpenAI)
│
├── 📁 skills/                   # 🧠 SKILLS (HABILIDADES)
│   ├── base_skill.py            # Classe base abstrata para Skills
│   ├── research_skill.py        # 🔍 Pesquisa aprofundada (web + wiki)
│   ├── summarize_skill.py       # 📋 Resumo inteligente de textos
│   └── content_skill.py         # ✉️ Criação de conteúdo (e-mail, relatório, post)
│
├── 📁 tools/                    # 🔧 FERRAMENTAS
│   ├── calculator.py            # Calculadora matemática
│   ├── datetime_tool.py         # Data e hora
│   ├── web_search.py            # Busca web (DuckDuckGo)
│   ├── rag_tool.py              # Busca no RAG
│   ├── geocoding.py             # Geocoding (Nominatim)
│   ├── crypto.py                # Criptomoedas (CoinGecko)
│   ├── stocks.py                # Ações/Forex (Alpha Vantage)
│   └── wikipedia.py             # Wikipedia API
│
├── 📁 knowledge_base/           # 📚 RAG
│   ├── document_loader.py       # Carregador de documentos
│   └── vector_store.py          # Vector Store (FAISS)
│
├── 📁 core/                     # 🧠 CORE
│   └── memory.py                # Sistema de memória
│
├── 📁 static/                   # 🎨 ARQUIVOS ESTÁTICOS
│   ├── chat_sse_demo.html       # Demo interativo
│   └── nginx.conf               # Config Nginx (Docker)
│
└── 📁 logs/                     # 📋 LOGS
    └── .gitkeep
```

---

## 🤖 Agentes Disponíveis

### Agentes Base

| ID | Nome | Provider | Especialização | Tools | RAG |
|----|------|----------|----------------|:-----:|:---:|
| `simple-openai` | Simple Agent | OpenAI | Geral (sem tools) | ❌ | ❌ |
| `simple-gemini` | Simple Agent | Google | Geral (sem tools) | ❌ | ❌ |
| `openai` | OpenAI Agent | OpenAI | Geral | ✅ | ✅ |
| `gemini` | Gemini Agent | Google | Geral | ✅ | ✅ |
| `azure` | Azure OpenAI Agent | Azure | Geral (compliance empresarial) | ✅ | ✅ |
| `ollama` | **Ollama Agent** | **Local** | Geral **(sem API key!)** | ✅ | ✅ |

### Agentes Especialistas

| ID | Nome | Provider | Especialização | Tools |
|----|------|----------|----------------|:-----:|
| `finance-openai` | Finance Expert | OpenAI | 💰 Finanças (ações, crypto, câmbio) | ✅ |
| `finance-gemini` | Finance Expert | Google | 💰 Finanças (ações, crypto, câmbio) | ✅ |
| `knowledge-openai` | Knowledge Expert | OpenAI | 📚 Conhecimento (Wikipedia) | ✅ |
| `knowledge-gemini` | Knowledge Expert | Google | 📚 Conhecimento (Wikipedia) | ✅ |
| `websearch-openai` | Web Search Expert | OpenAI | 🔍 Pesquisa Web (DuckDuckGo) | ✅ |
| `websearch-gemini` | Web Search Expert | Google | 🔍 Pesquisa Web (DuckDuckGo) | ✅ |
| `skills-azure` | **Skills Agent** | **Azure** | 🧠 **Skills: Pesquisa + Resumo + Criação** | ✅ |

### Agentes MCP (Model Context Protocol)

| ID | Nome | Descrição | Requer API Key? |
|----|------|-----------|:---------------:|
| `mcp-demo` | MCP Demo | Demonstração do conceito MCP (sem conexão real) | ❌ |
| `mcp-fetch` | MCP Fetch | Busca e extrai conteúdo de URLs da web | ❌ |
| `mcp-filesystem` | MCP Filesystem | Lê e escreve arquivos no sistema local | ❌ |
| `mcp-memory` | MCP Memory | Armazena e recupera informações na memória | ❌ |
| `mcp-time` | MCP Time | Informações de data, hora e fuso horário | ❌ |
| `mcp-sqlite` | MCP SQLite | Consultas em banco de dados SQLite | ❌ |
| `mcp-brave_search` | MCP Brave Search | Busca na web via Brave Search API | ⚠️ `BRAVE_API_KEY` |
| `mcp-github` | MCP GitHub | Acesso a repositórios e dados do GitHub | ⚠️ `GITHUB_TOKEN` |

> 🦙 **Ollama**: Roda modelos localmente, sem API key, com total privacidade!
>
> ☁️ **Azure OpenAI**: Mesmos modelos GPT, com compliance empresarial, SLA e rede privada.
>
> 🔌 **MCP**: Agentes MCP reais requerem Node.js (npx) instalado. Os que exigem API key só aparecem se a chave estiver configurada.

---

## 🧠 Skills (Habilidades)

### O que são Skills?

**Skills** são uma abstração de **nível superior às Tools**. Enquanto uma Tool executa uma ação atômica, uma Skill encapsula uma **capacidade completa** que pode compor múltiplas tools internamente.

```
┌─────────────┬───────────────────────────┬──────────────────────────────────┐
│             │ Tool                      │ Skill                            │
├─────────────┼───────────────────────────┼──────────────────────────────────┤
│ Escopo      │ Ação única e atômica      │ Capacidade completa e composta   │
│ Complexidade│ Simples (1 função)        │ Complexa (multi-etapa)           │
│ Composição  │ Independente              │ Pode usar múltiplas Tools        │
│ Exemplo     │ web_search("clima SP")    │ research("impacto climático SP") │
└─────────────┴───────────────────────────┴──────────────────────────────────┘
```

**Analogia do mundo real:**
- 🔨 **Tool** = Martelo, Serra, Chave de Fenda (ferramentas individuais)
- 🧠 **Skill** = Carpintaria, Encanamento (habilidades que usam várias ferramentas)

### Arquitetura de Skills

```
┌───────────────────────────────────────────────────────────────┐
│                        AGENTE (SkillsAgent)                   │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    SKILLS (Alto Nível)                  │  │
│  │                                                         │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │  │
│  │  │ 🔍 Research  │ │ 📋 Summarize │ │ ✉️ Content       │ │  │
│  │  │              │ │              │ │    Creation      │ │  │
│  │  │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────────┐ │ │  │
│  │  │ │web_search│ │ │ │calculator│ │ │ │datetime_tool │ │ │  │
│  │  │ │wikipedia │ │ │ │  regex   │ │ │ │  templates   │ │ │  │
│  │  │ └──────────┘ │ │ └──────────┘ │ │ └──────────────┘ │ │  │
│  │  └──────────────┘ └──────────────┘ └──────────────────┘ │  │
│  │                     TOOLS (Baixo Nível)                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  LLM: Azure OpenAI (GPT-4o)  │  Framework: LangGraph (ReAct)  │
└───────────────────────────────────────────────────────────────┘
```

### Skills Disponíveis

| Skill | Descrição | Tools Internas | Exemplo de Uso |
|-------|-----------|----------------|----------------|
| 🔍 `research_skill` | Pesquisa aprofundada com múltiplas fontes | `web_search`, `wikipedia` | "Pesquise sobre energia solar" |
| 📋 `summarize_skill` | Análise e resumo inteligente com métricas | `calculator`, `regex` | "Resuma este texto: ..." |
| ✉️ `content_creation_skill` | Criação de conteúdo profissional formatado | `datetime`, `templates` | "Escreva um e-mail sobre a reunião" |

### Detalhamento das Skills

#### 🔍 Research Skill
Combina busca web + Wikipedia e sintetiza em um relatório estruturado.
- **Profundidades:** `rapida` (só web), `normal` (web + wiki), `profunda` (web + wiki + detalhes)
- **Saída:** Relatório formatado com seções e fontes citadas

#### 📋 Summarize Skill
Analisa texto com métricas (palavras, complexidade) e gera resumos estruturados.
- **Estilos:** `executivo` (formal conciso), `pontos_chave` (bullet points), `analise_critica` (prós/contras)
- **Saída:** Resumo formatado com métricas e destaques

#### ✉️ Content Creation Skill
Gera conteúdo profissional usando templates especializados.
- **Tipos:** `email` (profissional), `report` (relatório), `social_post` (LinkedIn + Twitter/X)
- **Tons:** `formal`, `profissional`, `casual`, `entusiasmado`
- **Saída:** Conteúdo formatado pronto para uso

### Padrões de Projeto nas Skills

| Padrão | Onde é Usado | Benefício |
|--------|-------------|-----------|
| **Template Method** | `BaseSkill` (execute → validate → format) | Estrutura consistente |
| **Strategy** | `SummarizeSkill` (estilos de resumo) | Comportamentos intercambiáveis |
| **Factory** | `ContentCreationSkill` (tipos de conteúdo) | Criação flexível |
| **Composição** | Todas as Skills (compõem Tools) | Reutilização e modularidade |
| **Open/Closed** | `SkillsAgent.add_skill()` | Extensível sem modificar |

---

## 🔧 Tools (Ferramentas)

### Tools Disponíveis

| Tool | Função | Exemplo de Pergunta |
|------|--------|---------------------|
| 🧮 `calculator` | Cálculos matemáticos | "Quanto é 15% de 230?" |
| 📅 `get_current_datetime` | Data e hora atual | "Que dia é hoje?" |
| 🔍 `web_search` | Busca na web | "Pesquise sobre LangChain" |
| 📚 `knowledge_base_search` | Busca no RAG | "O que diz o documento?" |
| 🌍 `geocode_address` | Endereço → Coordenadas | "Coordenadas da Av. Paulista?" |
| 📍 `reverse_geocode` | Coordenadas → Endereço | "Que lugar é -23.55, -46.63?" |
| 🪙 `crypto_price` | Preço de criptomoeda | "Preço do Bitcoin?" |
| 🏆 `top_cryptos` | Ranking de cryptos | "Top 10 criptomoedas?" |
| 📊 `stock_quote` | Cotação de ações | "Preço da Apple?" |
| 💱 `forex_rate` | Taxa de câmbio | "Cotação do dólar?" |
| 📖 `wikipedia_summary` | Resumo da Wikipedia | "Quem foi Einstein?" |
| 🔎 `wikipedia_search` | Busca na Wikipedia | "Artigos sobre física quântica" |

### APIs Utilizadas (Gratuitas)

| Tool | API | Precisa de Key? |
|------|-----|-----------------|
| Busca Web | DuckDuckGo | ❌ Não |
| Geocoding | Nominatim/OSM | ❌ Não |
| Criptomoedas | CoinGecko | ❌ Não |
| Ações/Forex | Alpha Vantage | ⚠️ Gratuita |
| Wikipedia | Wikipedia API | ❌ Não |

---

## 🌐 API REST

### Endpoints Principais

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/agents` | Lista agentes disponíveis |
| `GET` | `/agents/{id}` | Detalhes de um agente |
| `POST` | `/sessions` | Cria sessão de chat |
| `GET` | `/sessions` | Lista sessões ativas |
| `POST` | `/chat/{session_id}` | Envia mensagem |
| `POST` | `/chat/{session_id}/stream` | **Chat com streaming** |
| `POST` | `/chat/quick/{agent_id}` | Chat rápido (sem sessão) |
| `POST` | `/chat/quick/{agent_id}/stream` | **Chat rápido com streaming** |
| `GET` | `/tools` | Lista ferramentas |
| `GET` | `/health` | Status da API |
| `GET` | `/demo` | **Página de demonstração** |

### Exemplo: Chat com Streaming (JavaScript)

```javascript
async function chat(message) {
    const response = await fetch('/chat/quick/openai/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        for (const line of chunk.split('\n')) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') return;
                
                const parsed = JSON.parse(data);
                if (parsed.type === 'token') {
                    console.log(parsed.content); // Token recebido!
                }
            }
        }
    }
}
```

### Exemplo: Chat com Streaming (Python)

```python
import requests

def chat(message: str):
    response = requests.post(
        "http://localhost:8000/chat/quick/openai/stream",
        json={"message": message},
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break
                
                import json
                parsed = json.loads(data)
                if parsed['type'] == 'token':
                    print(parsed['content'], end='', flush=True)
    print()

chat("Explique o que é machine learning")
```

### Exemplo: Usando cURL

```bash
# Listar agentes
curl http://localhost:8000/agents

# Criar sessão
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "openai"}'

# Chat rápido
curl -X POST http://localhost:8000/chat/quick/openai \
  -H "Content-Type: application/json" \
  -d '{"message": "Olá, tudo bem?"}'
```

---

## 🎮 Demo Interativo

Acesse **http://localhost:8000/demo** para usar o chat interativo com:

### ✨ Features do Demo

- 💬 **Chat em tempo real** com streaming SSE
- 🎨 **3 Temas**: Default, ChatGPT, Gemini
- 📝 **Histórico de conversas** persistente
- 📊 **Contagem de tokens** (input/output)
- ⚙️ **Configurações** de agente, modelo e temperatura
- 📱 **Responsivo** para mobile

### 🎨 Temas Disponíveis

| Tema | Descrição                            |
|------|--------------------------------------|
| 🌙 Default | Tema escuro com gradiente roxo/ciano |
| 💚 ChatGPT | Similar ao ChatGPT da OpenAI         |
| 💙 Gemini | Similar ao Google Gemini             |

### 📸 Screenshots

<p align="center">
  <img src="assets/default.png" alt="Tema Default" width="400"/>
  <br><em>🌙 Tema Default</em>
</p>

<p align="center">
  <img src="assets/openai.png" alt="Tema ChatGPT" width="400"/>
  <br><em>💚 Tema ChatGPT (OpenAI)</em>
</p>

<p align="center">
  <img src="assets/gemini.png" alt="Tema Gemini" width="400"/>
  <br><em>💙 Tema Gemini (Google)</em>
</p>

---

## 📚 Conceitos Importantes

### 🤖 O que é um Agente?

Um **Agente de IA** é um programa que:

```
┌─────────────────────────────────────────────────────────┐
│                      AGENTE DE IA                       │
├─────────────────────────────────────────────────────────┤
│  1. ENTENDE → Analisa a mensagem do usuário             │
│  2. DECIDE  → Escolhe qual ação tomar                   │
│  3. EXECUTA → Usa tools, RAG ou responde diretamente    │
│  4. FORMULA → Gera resposta baseada no resultado        │
└─────────────────────────────────────────────────────────┘
```

### 🔧 O que são Tools?

**Tools** são funções que o agente pode chamar quando necessário:

```python
@tool("calculator")
def calculator(expression: str) -> str:
    """Calcula expressões matemáticas."""
    return str(eval(expression))

# O LLM decide QUANDO usar:
# "Quanto é 10 + 20?" → Usa calculator
# "Olá, tudo bem?"    → Não usa (responde direto)
```

### 🧠 O que são Skills?

**Skills** são capacidades de **alto nível** que vão além de Tools simples:

```
┌─────────────────────────────────────────────────────────────┐
│                          SKILL                              │
├─────────────────────────────────────────────────────────────┤
│  1. RECEBE  → Tarefa complexa do agente                     │
│  2. COMPÕE  → Usa múltiplas tools internamente              │
│  3. PROCESSA→ Lógica multi-etapa (pipeline)                 │
│  4. ENTREGA → Resultado estruturado e completo              │
└─────────────────────────────────────────────────────────────┘
```

```python
# Uma Skill COMPÕE múltiplas tools:
class ResearchSkill(BaseSkill):
    def execute(self, topic: str) -> str:
        # Etapa 1: Busca na web
        web_results = web_search(topic)
        # Etapa 2: Busca na Wikipedia
        wiki_results = wikipedia(topic)
        # Etapa 3: Sintetiza tudo
        report = synthesize(web_results, wiki_results)
        return report

# O LLM decide qual SKILL usar:
# "Pesquise sobre IA"        → Usa research_skill
# "Resuma este texto: ..."   → Usa summarize_skill
# "Escreva um e-mail"        → Usa content_creation_skill
```

### 📚 O que é RAG?

**RAG** (Retrieval Augmented Generation) dá conhecimento específico ao LLM:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  DOCUMENTOS  │ →  │   VETORES    │ →  │    BUSCA     │
│  PDF, DOCX   │    │   FAISS      │    │  Relevantes  │
└──────────────┘    └──────────────┘    └──────────────┘
                                               ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   RESPOSTA   │ ←  │     LLM      │ ←  │   CONTEXTO   │
│   Precisa    │    │  GPT/Gemini  │    │  + Pergunta  │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 🧠 Tipos de Memória

| Tipo | Descrição | Persistência |
|------|-----------|--------------|
| **Sem Memória** | Cada mensagem é independente | ❌ |
| **Curto Prazo** | Últimas N mensagens | Sessão |
| **Longo Prazo** | Fatos importantes | Disco |
| **Combinada** | Curto + Longo prazo | Ambos |

---

## 🛠️ Criando Seus Próprios Componentes

### Criando um Novo Agente

```python
# agents/meu_agente.py
from agents.base_agent import BaseAgent
from langchain_openai import ChatOpenAI

class MeuAgente(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Meu Agente",
            description="Um agente personalizado"
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini")
    
    def process_message(self, message: str) -> str:
        response = self.llm.invoke(message)
        return response.content
```

### Criando uma Nova Tool

```python
# tools/minha_tool.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class MinhaToolInput(BaseModel):
    query: str = Field(description="O que buscar")

@tool("minha_tool", args_schema=MinhaToolInput)
def minha_tool(query: str) -> str:
    """
    Descrição da tool para o LLM saber quando usar.
    
    Use quando o usuário perguntar sobre X.
    """
    # Sua lógica aqui
    resultado = fazer_algo(query)
    return resultado
```

### Criando uma Nova Skill

```python
# skills/minha_skill.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from skills.base_skill import BaseSkill

class MinhaSkillInput(BaseModel):
    topic: str = Field(description="O tópico a processar")
    depth: str = Field(default="normal", description="Profundidade")

class MinhaSkill(BaseSkill):
    def __init__(self):
        super().__init__(
            name="minha_skill",
            description="Faz algo incrível com múltiplas etapas",
            required_tools=["web_search", "calculator"]
        )

    def execute(self, topic: str, depth: str = "normal") -> str:
        # Etapa 1: Coleta dados
        dados = self._coletar(topic)
        # Etapa 2: Processa
        resultado = self._processar(dados, depth)
        # Etapa 3: Formata
        return self._formatar(resultado)

# Exporta como LangChain Tool
_skill = MinhaSkill()

@tool("minha_skill", args_schema=MinhaSkillInput)
def minha_skill_tool(topic: str, depth: str = "normal") -> str:
    """Descrição da skill para o LLM."""
    return _skill.execute(topic=topic, depth=depth)
```

### Registrando no Sistema

```python
# api.py - Adicione no agent_registry
agent_registry.register(
    agent_id="meu-agente",
    config={
        "name": "Meu Agente",
        "class": "MeuAgente",
        "provider": "openai",
        # ...
    }
)
```

---

## 🐳 Docker

O projeto inclui suporte completo a Docker com **docker-compose** para subir a API e o Chat UI de forma isolada.

### Serviços

| Serviço | Descrição | Porta Padrão | Imagem |
|---------|-----------|:------------:|--------|
| `api` | API FastAPI com os Agentes | `8000` | Build local (`Dockerfile`) |
| `chat-ui` | Chat SSE Demo (Nginx) | `8080` | `nginx:alpine` |

### Quick Start com Docker

```bash
# 1. Configure as variáveis de ambiente
cp .env.example .env
nano .env   # Adicione suas API keys

# 2. Suba tudo
docker-compose up -d --build

# 3. Acompanhe os logs
docker-compose logs -f
```

### URLs dos Serviços

| Interface | URL |
|-----------|-----|
| 🎮 **Chat UI** | http://localhost:8080 |
| 📚 **API Docs** | http://localhost:8000/docs |
| ❤️ **Health Check** | http://localhost:8000/health |

### Personalização de Portas

As portas podem ser alteradas via variáveis de ambiente no `.env`:

```env
API_PORT=8000         # Porta da API FastAPI
CHAT_UI_PORT=8080     # Porta do Chat UI (Nginx)
```

### Comandos Úteis

```bash
# Subir em background
docker-compose up -d --build

# Ver logs em tempo real
docker-compose logs -f

# Ver logs de um serviço específico
docker-compose logs -f api
docker-compose logs -f chat-ui

# Parar e remover
docker-compose down

# Rebuild após alterações
docker-compose up -d --build --force-recreate
```

### Arquitetura Docker

```
┌──────────────────────────────────────────────────┐
│                 docker-compose                   │
│                                                  │
│  ┌──────────────┐       ┌──────────────────┐     │
│  │   chat-ui    │       │      api         │     │
│  │  (nginx)     │──────▶│   (FastAPI)      │     │
│  │  :8080       │       │   :8000          │     │
│  └──────────────┘       └──────────────────┘     │
│                                                  │
│              genai-network (bridge)              │
└──────────────────────────────────────────────────┘
```

> 💡 O serviço `chat-ui` só inicia após a API passar no health check, garantindo que o backend esteja pronto.

---

## 🔑 Configuração

### Variáveis de Ambiente

```env
# === LLM APIs (pelo menos uma obrigatória) ===
OPENAI_API_KEY=sk-sua-chave-aqui
GOOGLE_API_KEY=sua-chave-aqui

# === Azure OpenAI (para Skills Agent e Azure Agent) ===
AZURE_OPENAI_API_KEY=sua-chave-azure-aqui
AZURE_OPENAI_ENDPOINT=https://seu-recurso.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# === APIs de Tools (opcionais) ===
ALPHA_VANTAGE_API_KEY=sua-chave  # Para ações/forex

# === API Config (opcionais) ===
API_PORT=8000
API_HOST=0.0.0.0
CHAT_UI_PORT=8080
API_AUTH_REQUIRED=false
API_AUTH_KEY=sua-chave-secreta
```

### Onde Obter as API Keys

| API | URL | Custo |
|-----|-----|-------|
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | Pago |
| Google AI | [aistudio.google.com](https://aistudio.google.com/apikey) | Gratuito |
| Azure OpenAI | [portal.azure.com](https://portal.azure.com) | Pago (Azure) |
| Alpha Vantage | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | Gratuito |

---

## 📖 Exemplos de Uso

### Exemplo 1: Chat Simples

```python
from agents import SimpleOpenAIAgent

agent = SimpleOpenAIAgent()
response = agent.process_message("Olá, tudo bem?")
print(response)
```

### Exemplo 2: Agente com Tools

```python
from agents import OpenAIAgent

agent = OpenAIAgent()
response = agent.process_message("Quanto é 15% de 350?")
print(response)  # Usa a calculadora automaticamente
```

### Exemplo 3: Agente de Finanças

```python
from agents import FinanceOpenAIAgent

agent = FinanceOpenAIAgent()
response = agent.process_message("Qual o preço do Bitcoin?")
print(response)  # Usa a API CoinGecko
```

### Exemplo 4: Agente com Skills (Azure OpenAI)

```python
from agents import SkillsAgent

# Cria o agente com Skills (requer AZURE_OPENAI_API_KEY e AZURE_OPENAI_ENDPOINT)
agent = SkillsAgent()

# Research Skill: pesquisa aprofundada com múltiplas fontes
response = agent.process_message("Pesquise sobre inteligência artificial generativa")
print(response)  # Relatório com dados da web + Wikipedia

# Summarize Skill: resume textos com métricas
response = agent.process_message("Resuma este texto: A IA está revolucionando...")
print(response)  # Resumo estruturado com pontos-chave

# Content Creation Skill: cria conteúdo profissional
response = agent.process_message("Escreva um e-mail formal sobre a reunião de projeto")
print(response)  # E-mail formatado com template profissional

# Verificar skills e tools disponíveis
print(agent.list_skills())  # ['research_skill', 'summarize_skill', 'content_creation_skill']
print(agent.list_tools())   # Skills + calculator + datetime
```

### Exemplo 5: Usando a API

```python
import requests

# Criar sessão
session = requests.post("http://localhost:8000/sessions", json={
    "agent_id": "finance-openai"
}).json()

# Enviar mensagens
response = requests.post(
    f"http://localhost:8000/chat/{session['session_id']}",
    json={"message": "Cotação do dólar hoje?"}
).json()

print(response['response'])
```

---

## 🧰 Comandos Úteis (Makefile)

```bash
# Instalação
make install          # Instala dependências
make install-dev      # Instala com deps de desenvolvimento

# Execução
make dev              # Inicia API + Streamlit (desenvolvimento)
make api              # Inicia apenas a API
make app              # Inicia apenas o Streamlit

# Background
make start            # Inicia tudo em background
make stop             # Para todos os serviços
make restart          # Reinicia tudo
make status           # Verifica status

# Logs
make logs             # Mostra logs recentes
make logs-api         # Logs da API em tempo real
make logs-app         # Logs do Streamlit em tempo real

# Qualidade
make test             # Executa testes
make lint             # Verifica código
make format           # Formata código

# Utilidades
make clean            # Limpa arquivos temporários
make check-env        # Verifica variáveis de ambiente
make info             # Informações do projeto
make help             # Lista todos os comandos

# Docker
docker-compose up -d --build    # Sobe API + Chat UI
docker-compose down             # Para e remove tudo
docker-compose logs -f          # Logs em tempo real
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Add MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 🙏 Agradecimentos

- [LangChain](https://langchain.com) - Framework de LLM
- [FastAPI](https://fastapi.tiangolo.com) - API Framework
- [Streamlit](https://streamlit.io) - Interface Web
- [OpenAI](https://openai.com) - GPT Models
- [Google AI](https://ai.google.dev) - Gemini Models

---

<div align="center">

**GenAI Master**

🎓 2026

</div>

