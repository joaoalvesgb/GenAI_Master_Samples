# ============================================
# GenAI Master Samples - Makefile
# ============================================
# Facilita a execução de comandos comuns do projeto
# Use: make help para ver todos os comandos disponíveis
# ============================================

.PHONY: help install dev api app clean test lint format docker-build docker-run docker-stop logs k8s-deploy k8s-dry-run k8s-delete k8s-status k8s-build k8s-forward

# Cores para output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Variáveis
PYTHON := poetry run python
UVICORN := poetry run uvicorn
STREAMLIT := poetry run streamlit
PORT_API := 8000
PORT_APP := 8501

# ============================================
# HELP
# ============================================

help: ## Mostra esta mensagem de ajuda
	@echo ""
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║        🤖 GenAI Master Samples - Comandos Disponíveis       ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Exemplos de uso:$(NC)"
	@echo "  make install    - Instala todas as dependências"
	@echo "  make dev        - Inicia API + Streamlit em modo desenvolvimento"
	@echo "  make api        - Inicia apenas a API FastAPI"
	@echo "  make app        - Inicia apenas o Streamlit"
	@echo ""

# ============================================
# INSTALAÇÃO
# ============================================

install: ## Instala todas as dependências do projeto
	@echo "$(BLUE)📦 Instalando dependências...$(NC)"
	poetry install
	@echo "$(GREEN)✅ Dependências instaladas com sucesso!$(NC)"

install-dev: ## Instala dependências incluindo dev
	@echo "$(BLUE)📦 Instalando dependências de desenvolvimento...$(NC)"
	poetry install --with dev
	@echo "$(GREEN)✅ Dependências de dev instaladas!$(NC)"

update: ## Atualiza todas as dependências
	@echo "$(BLUE)🔄 Atualizando dependências...$(NC)"
	poetry update
	@echo "$(GREEN)✅ Dependências atualizadas!$(NC)"

# ============================================
# EXECUÇÃO - DESENVOLVIMENTO
# ============================================

dev: ## Inicia API e Streamlit em modo desenvolvimento (paralelo)
	@echo "$(BLUE)🚀 Iniciando ambiente de desenvolvimento...$(NC)"
	@echo "$(YELLOW)API:$(NC) http://localhost:$(PORT_API)"
	@echo "$(YELLOW)App:$(NC) http://localhost:$(PORT_APP)"
	@echo "$(YELLOW)Demo:$(NC) http://localhost:$(PORT_API)/demo"
	@echo ""
	@make -j2 api app

api: ## Inicia a API FastAPI (porta 8000)
	@echo "$(BLUE)🔌 Iniciando API FastAPI...$(NC)"
	@echo "$(GREEN)➜$(NC) http://localhost:$(PORT_API)"
	@echo "$(GREEN)➜$(NC) Docs: http://localhost:$(PORT_API)/docs"
	@echo "$(GREEN)➜$(NC) Demo: http://localhost:$(PORT_API)/demo"
	$(UVICORN) api:app --host 0.0.0.0 --port $(PORT_API) --reload

api-prod: ## Inicia a API em modo produção
	@echo "$(BLUE)🔌 Iniciando API em produção...$(NC)"
	$(UVICORN) api:app --host 0.0.0.0 --port $(PORT_API) --workers 4

app: ## Inicia o Streamlit App (porta 8501)
	@echo "$(BLUE)🎨 Iniciando Streamlit App...$(NC)"
	@echo "$(GREEN)➜$(NC) http://localhost:$(PORT_APP)"
	$(STREAMLIT) run app.py --server.port $(PORT_APP)

# ============================================
# EXECUÇÃO - BACKGROUND
# ============================================

start: ## Inicia API e Streamlit em background
	@echo "$(BLUE)🚀 Iniciando serviços em background...$(NC)"
	@make start-api
	@make start-app
	@echo ""
	@echo "$(GREEN)✅ Serviços iniciados!$(NC)"
	@echo "$(YELLOW)API:$(NC) http://localhost:$(PORT_API)"
	@echo "$(YELLOW)App:$(NC) http://localhost:$(PORT_APP)"
	@echo "$(YELLOW)Demo:$(NC) http://localhost:$(PORT_API)/demo"
	@echo ""
	@echo "Use $(YELLOW)make stop$(NC) para parar os serviços"
	@echo "Use $(YELLOW)make logs$(NC) para ver os logs"

start-api: ## Inicia a API em background
	@echo "$(BLUE)🔌 Iniciando API em background...$(NC)"
	@nohup $(UVICORN) api:app --host 0.0.0.0 --port $(PORT_API) > logs/api.log 2>&1 &
	@echo "$(GREEN)✅ API iniciada (PID: $$!)$(NC)"

start-app: ## Inicia o Streamlit em background
	@echo "$(BLUE)🎨 Iniciando Streamlit em background...$(NC)"
	@mkdir -p logs
	@nohup $(STREAMLIT) run app.py --server.port $(PORT_APP) > logs/streamlit.log 2>&1 &
	@echo "$(GREEN)✅ Streamlit iniciado (PID: $$!)$(NC)"

stop: ## Para todos os serviços
	@echo "$(RED)🛑 Parando serviços...$(NC)"
	@pkill -f "uvicorn api:app" 2>/dev/null || true
	@pkill -f "streamlit run app.py" 2>/dev/null || true
	@echo "$(GREEN)✅ Serviços parados!$(NC)"

stop-api: ## Para apenas a API
	@echo "$(RED)🛑 Parando API...$(NC)"
	@pkill -f "uvicorn api:app" 2>/dev/null || true
	@echo "$(GREEN)✅ API parada!$(NC)"

stop-app: ## Para apenas o Streamlit
	@echo "$(RED)🛑 Parando Streamlit...$(NC)"
	@pkill -f "streamlit run app.py" 2>/dev/null || true
	@echo "$(GREEN)✅ Streamlit parado!$(NC)"

restart: stop start ## Reinicia todos os serviços

# ============================================
# LOGS
# ============================================

logs: ## Mostra logs da API e Streamlit
	@echo "$(BLUE)📋 Logs da API:$(NC)"
	@tail -50 logs/api.log 2>/dev/null || echo "Nenhum log encontrado"
	@echo ""
	@echo "$(BLUE)📋 Logs do Streamlit:$(NC)"
	@tail -50 logs/streamlit.log 2>/dev/null || echo "Nenhum log encontrado"

logs-api: ## Mostra logs da API (tempo real)
	@echo "$(BLUE)📋 Logs da API (Ctrl+C para sair):$(NC)"
	@tail -f logs/api.log

logs-app: ## Mostra logs do Streamlit (tempo real)
	@echo "$(BLUE)📋 Logs do Streamlit (Ctrl+C para sair):$(NC)"
	@tail -f logs/streamlit.log

# ============================================
# TESTES E QUALIDADE
# ============================================

test: ## Executa os testes
	@echo "$(BLUE)🧪 Executando testes...$(NC)"
	$(PYTHON) -m pytest tests/ -v
	@echo "$(GREEN)✅ Testes concluídos!$(NC)"

test-cov: ## Executa testes com cobertura
	@echo "$(BLUE)🧪 Executando testes com cobertura...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html
	@echo "$(GREEN)✅ Relatório de cobertura gerado em htmlcov/$(NC)"

lint: ## Verifica código com ruff
	@echo "$(BLUE)🔍 Verificando código...$(NC)"
	poetry run ruff check .
	@echo "$(GREEN)✅ Verificação concluída!$(NC)"

format: ## Formata código com ruff
	@echo "$(BLUE)✨ Formatando código...$(NC)"
	poetry run ruff format .
	@echo "$(GREEN)✅ Código formatado!$(NC)"

type-check: ## Verifica tipos com mypy
	@echo "$(BLUE)🔍 Verificando tipos...$(NC)"
	poetry run mypy .
	@echo "$(GREEN)✅ Verificação de tipos concluída!$(NC)"

# ============================================
# DOCKER
# ============================================

docker-build: ## Constrói imagem Docker
	@echo "$(BLUE)🐳 Construindo imagem Docker...$(NC)"
	docker build -t genai-api:latest .
	@echo "$(GREEN)✅ Imagem construída!$(NC)"

docker-run: ## Executa container Docker (standalone)
	@echo "$(BLUE)🐳 Iniciando container...$(NC)"
	docker run -d --name genai-api \
		-p $(PORT_API):8000 \
		--env-file .env \
		genai-api:latest
	@echo "$(GREEN)✅ Container iniciado!$(NC)"
	@echo "$(YELLOW)API:$(NC) http://localhost:$(PORT_API)"
	@echo "$(YELLOW)Demo:$(NC) http://localhost:$(PORT_API)/demo"

docker-stop: ## Para container Docker
	@echo "$(RED)🛑 Parando container...$(NC)"
	docker stop genai-api 2>/dev/null || true
	docker rm genai-api 2>/dev/null || true
	@echo "$(GREEN)✅ Container parado!$(NC)"

docker-logs: ## Mostra logs do container
	docker logs -f genai-api

docker-shell: ## Acessa shell do container
	docker exec -it genai-api /bin/bash

# Docker Compose
docker-up: ## Inicia com docker-compose
	@echo "$(BLUE)🐳 Iniciando com docker-compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✅ Serviços iniciados!$(NC)"
	@echo "$(YELLOW)API:$(NC) http://localhost:$(PORT_API)"
	@echo "$(YELLOW)Demo:$(NC) http://localhost:$(PORT_API)/demo"

docker-down: ## Para docker-compose
	@echo "$(RED)🛑 Parando docker-compose...$(NC)"
	docker-compose down
	@echo "$(GREEN)✅ Serviços parados!$(NC)"

docker-rebuild: ## Reconstrói e reinicia com docker-compose
	@echo "$(BLUE)🐳 Reconstruindo...$(NC)"
	docker-compose up -d --build
	@echo "$(GREEN)✅ Serviços reiniciados!$(NC)"

docker-ps: ## Lista containers em execução
	docker-compose ps

# ============================================
# UTILIDADES
# ============================================

clean: ## Limpa arquivos temporários
	@echo "$(BLUE)🧹 Limpando arquivos temporários...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/*.log 2>/dev/null || true
	@echo "$(GREEN)✅ Limpeza concluída!$(NC)"

clean-all: clean ## Limpa tudo incluindo venv
	@echo "$(BLUE)🧹 Limpando tudo...$(NC)"
	rm -rf .venv 2>/dev/null || true
	@echo "$(GREEN)✅ Limpeza completa!$(NC)"

shell: ## Abre shell Python com contexto do projeto
	@echo "$(BLUE)🐍 Abrindo shell Python...$(NC)"
	$(PYTHON)

check-env: ## Verifica variáveis de ambiente
	@echo "$(BLUE)🔐 Verificando variáveis de ambiente...$(NC)"
	@echo ""
	@if [ -f .env ]; then \
		echo "$(GREEN)✅ Arquivo .env encontrado$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Variáveis configuradas:$(NC)"; \
		grep -v "^#" .env | grep -v "^$$" | cut -d= -f1 | while read var; do \
			echo "  ✓ $$var"; \
		done; \
	else \
		echo "$(RED)❌ Arquivo .env não encontrado$(NC)"; \
		echo "$(YELLOW)Crie um arquivo .env baseado no .env.example$(NC)"; \
	fi
	@echo ""

status: ## Mostra status dos serviços
	@echo "$(BLUE)📊 Status dos serviços:$(NC)"
	@echo ""
	@if pgrep -f "uvicorn api:app" > /dev/null; then \
		echo "  $(GREEN)●$(NC) API FastAPI: $(GREEN)Rodando$(NC) (http://localhost:$(PORT_API))"; \
	else \
		echo "  $(RED)●$(NC) API FastAPI: $(RED)Parada$(NC)"; \
	fi
	@if pgrep -f "streamlit run app.py" > /dev/null; then \
		echo "  $(GREEN)●$(NC) Streamlit:   $(GREEN)Rodando$(NC) (http://localhost:$(PORT_APP))"; \
	else \
		echo "  $(RED)●$(NC) Streamlit:   $(RED)Parada$(NC)"; \
	fi
	@echo ""

info: ## Mostra informações do projeto
	@echo ""
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║              🤖 GenAI Master Samples                        ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)📁 Estrutura:$(NC)"
	@echo "  • agents/     - Agentes de IA (OpenAI, Gemini)"
	@echo "  • tools/      - Ferramentas (Calculator, Web Search, etc)"
	@echo "  • core/       - Componentes core (Memory)"
	@echo "  • knowledge_base/ - RAG e Vector Store"
	@echo "  • static/     - Arquivos estáticos (Demo HTML)"
	@echo ""
	@echo "$(YELLOW)🔗 URLs:$(NC)"
	@echo "  • API:        http://localhost:$(PORT_API)"
	@echo "  • Docs:       http://localhost:$(PORT_API)/docs"
	@echo "  • Demo Chat:  http://localhost:$(PORT_API)/demo"
	@echo "  • Streamlit:  http://localhost:$(PORT_APP)"
	@echo ""
	@echo "$(YELLOW)📚 Comandos úteis:$(NC)"
	@echo "  • make dev    - Inicia tudo em modo desenvolvimento"
	@echo "  • make start  - Inicia tudo em background"
	@echo "  • make stop   - Para todos os serviços"
	@echo "  • make status - Verifica status dos serviços"
	@echo ""

# ============================================
# KUBERNETES
# ============================================

k8s-deploy: ## Deploy no Kubernetes
	@./k8s/deploy.sh

k8s-dry-run: ## Mostra o que seria aplicado no K8s sem executar
	@./k8s/deploy.sh --dry-run

k8s-delete: ## Remove todos os recursos do K8s
	@./k8s/deploy.sh --delete

k8s-status: ## Verifica status do deploy no K8s
	@./k8s/deploy.sh --status

k8s-build: ## Build da imagem Docker + deploy no K8s
	@./k8s/deploy.sh --build

k8s-forward: ## Abre port-forward local (API:8000, UI:8080)
	@./k8s/deploy.sh --port-forward

# ============================================
# ATALHOS
# ============================================

run: dev ## Alias para 'make dev'
up: start ## Alias para 'make start'
down: stop ## Alias para 'make stop'
i: install ## Alias para 'make install'
s: status ## Alias para 'make status'
h: help ## Alias para 'make help'

