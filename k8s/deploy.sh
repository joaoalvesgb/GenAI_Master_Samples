#!/usr/bin/env bash
# ============================================
# GenAI Master Samples - Deploy no Kubernetes
# ============================================
#
# Uso:
#   ./deploy.sh                    # Deploy completo
#   ./deploy.sh --dry-run          # Apenas mostra o que seria aplicado
#   ./deploy.sh --delete           # Remove todos os recursos
#   ./deploy.sh --status           # Verifica status do deploy
#   ./deploy.sh --build            # Build da imagem + deploy
#   ./deploy.sh --port-forward     # Abre port-forward local
#
# Pré-requisitos:
#   - kubectl configurado e conectado a um cluster
#   - Docker (se usar --build)
# ============================================

set -euo pipefail

# ============================================
# Configurações
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$SCRIPT_DIR"
NAMESPACE="genai"
IMAGE_NAME="genai-api"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"  # Ex: docker.io/meuusuario, ghcr.io/meuusuario

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# Funções auxiliares
# ============================================
log_info()    { echo -e "${BLUE}ℹ ${NC} $*"; }
log_success() { echo -e "${GREEN}✅${NC} $*"; }
log_warn()    { echo -e "${YELLOW}⚠️ ${NC} $*"; }
log_error()   { echo -e "${RED}❌${NC} $*"; }

check_prerequisites() {
    log_info "Verificando pré-requisitos..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl não encontrado. Instale: https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl não está conectado a um cluster."
        exit 1
    fi

    log_success "Pré-requisitos OK"
    echo ""
}

check_secrets() {
    log_info "Verificando secrets..."

    local secrets_file="$K8S_DIR/secrets.yaml"
    if grep -q 'OPENAI_API_KEY: ""' "$secrets_file" && grep -q 'GOOGLE_API_KEY: ""' "$secrets_file"; then
        log_warn "As chaves de API no secrets.yaml estão vazias!"
        log_warn "Preencha pelo menos OPENAI_API_KEY ou GOOGLE_API_KEY antes do deploy."
        echo ""
        echo "  Encode suas chaves em base64:"
        echo "    echo -n 'sua-chave-openai' | base64"
        echo "    echo -n 'sua-chave-google' | base64"
        echo ""
        read -rp "Deseja continuar mesmo assim? (y/N) " response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Deploy cancelado."
            exit 0
        fi
    fi
}

# ============================================
# Build da imagem Docker
# ============================================
build_image() {
    log_info "Fazendo build da imagem Docker..."

    local full_image="${IMAGE_NAME}:${IMAGE_TAG}"
    if [[ -n "$REGISTRY" ]]; then
        full_image="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    fi

    docker build -t "$full_image" "$PROJECT_DIR"
    log_success "Imagem criada: $full_image"

    if [[ -n "$REGISTRY" ]]; then
        log_info "Fazendo push para registry..."
        docker push "$full_image"
        log_success "Push concluído: $full_image"

        # Atualiza a imagem no deployment
        log_info "Atualizando imagem no deployment..."
        kubectl set image deployment/genai-api api="$full_image" -n "$NAMESPACE" 2>/dev/null || true
    fi

    echo ""
}

# ============================================
# Deploy
# ============================================
deploy() {
    local dry_run="${1:-false}"

    log_info "Iniciando deploy no Kubernetes..."
    echo ""

    if [[ "$dry_run" == "true" ]]; then
        log_info "[DRY-RUN] Mostrando o que seria aplicado:"
        echo ""
        kubectl apply -k "$K8S_DIR" --dry-run=client
    else
        # 1. Criar namespace primeiro
        log_info "1/4 - Criando namespace..."
        kubectl apply -f "$K8S_DIR/namespace.yaml"

        # 2. Aplicar configurações (secrets, configmaps, service account)
        log_info "2/4 - Aplicando configurações..."
        kubectl apply -f "$K8S_DIR/serviceaccount.yaml"
        kubectl apply -f "$K8S_DIR/secrets.yaml"
        kubectl apply -f "$K8S_DIR/configmap.yaml"
        kubectl apply -f "$K8S_DIR/chat-ui-configmap.yaml"

        # 3. Criar PVCs e serviços
        log_info "3/4 - Criando volumes e serviços..."
        kubectl apply -f "$K8S_DIR/pvc.yaml"
        kubectl apply -f "$K8S_DIR/api-service.yaml"
        kubectl apply -f "$K8S_DIR/chat-ui-service.yaml"

        # 4. Deploy das aplicações
        log_info "4/4 - Fazendo deploy das aplicações..."
        kubectl apply -f "$K8S_DIR/api-deployment.yaml"
        kubectl apply -f "$K8S_DIR/chat-ui-deployment.yaml"
        kubectl apply -f "$K8S_DIR/api-hpa.yaml"
        kubectl apply -f "$K8S_DIR/networkpolicy.yaml"
    fi

    echo ""
    log_success "Deploy concluído!"
    echo ""
    show_status
}

# ============================================
# Delete
# ============================================
delete_all() {
    log_warn "Removendo TODOS os recursos do namespace '$NAMESPACE'..."
    read -rp "Tem certeza? (y/N) " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Cancelado."
        exit 0
    fi

    kubectl delete namespace "$NAMESPACE" --ignore-not-found
    log_success "Namespace '$NAMESPACE' e todos os recursos foram removidos."
}

# ============================================
# Status
# ============================================
show_status() {
    log_info "Status do deploy no namespace '$NAMESPACE':"
    echo ""

    echo "━━━ Pods ━━━"
    kubectl get pods -n "$NAMESPACE" -o wide 2>/dev/null || log_warn "Namespace '$NAMESPACE' não encontrado"
    echo ""

    echo "━━━ Services ━━━"
    kubectl get services -n "$NAMESPACE" 2>/dev/null || true
    echo ""

    echo "━━━ Deployments ━━━"
    kubectl get deployments -n "$NAMESPACE" 2>/dev/null || true
    echo ""

    echo "━━━ HPA ━━━"
    kubectl get hpa -n "$NAMESPACE" 2>/dev/null || true
    echo ""

    echo "━━━ PVCs ━━━"
    kubectl get pvc -n "$NAMESPACE" 2>/dev/null || true
    echo ""

    echo "━━━ Ingress ━━━"
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || true
    echo ""
}

# ============================================
# Port Forward
# ============================================
port_forward() {
    log_info "Iniciando port-forward..."
    echo ""
    echo "  API:     http://localhost:8000"
    echo "  Chat UI: http://localhost:8080"
    echo ""
    echo "  Pressione Ctrl+C para parar"
    echo ""

    # Inicia port-forward em background
    kubectl port-forward svc/genai-api 8000:8000 -n "$NAMESPACE" &
    local api_pid=$!

    kubectl port-forward svc/genai-chat-ui 8080:80 -n "$NAMESPACE" &
    local ui_pid=$!

    # Captura Ctrl+C para cleanup
    trap "kill $api_pid $ui_pid 2>/dev/null; echo ''; log_info 'Port-forward encerrado.'; exit 0" INT TERM

    wait
}

# ============================================
# Main
# ============================================
main() {
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║   GenAI Master Samples - K8s Deploy      ║"
    echo "╚══════════════════════════════════════════╝"
    echo ""

    check_prerequisites

    case "${1:-}" in
        --dry-run)
            deploy "true"
            ;;
        --delete)
            delete_all
            ;;
        --status)
            show_status
            ;;
        --build)
            build_image
            deploy "false"
            ;;
        --port-forward)
            port_forward
            ;;
        --help|-h)
            echo "Uso: $0 [opção]"
            echo ""
            echo "Opções:"
            echo "  (sem opção)      Deploy completo"
            echo "  --dry-run        Mostra o que seria aplicado sem executar"
            echo "  --delete         Remove todos os recursos"
            echo "  --status         Verifica status do deploy"
            echo "  --build          Build da imagem Docker + deploy"
            echo "  --port-forward   Abre port-forward local (API:8000, UI:8080)"
            echo "  --help           Mostra esta ajuda"
            echo ""
            echo "Variáveis de ambiente:"
            echo "  IMAGE_TAG        Tag da imagem (default: latest)"
            echo "  REGISTRY         Registry Docker (ex: ghcr.io/usuario)"
            echo ""
            ;;
        "")
            check_secrets
            deploy "false"
            ;;
        *)
            log_error "Opção desconhecida: $1"
            echo "Use --help para ver as opções disponíveis."
            exit 1
            ;;
    esac
}

main "$@"

