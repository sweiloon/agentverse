#!/bin/bash

# AgentVerse Production Deployment Script
# Usage: ./deploy.sh [command]
# Commands: setup, deploy, update, logs, stop, restart, backup

set -e

# Configuration
PROJECT_NAME="agentverse"
DEPLOY_DIR="/opt/agentverse"
COMPOSE_FILE="docker-compose.prod.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Installing..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        log_info "Docker installed. Please log out and back in, then run this script again."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Installing..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi

    log_info "Docker and Docker Compose are available"
}

# Initial setup
setup() {
    log_info "Setting up AgentVerse production environment..."

    # Check Docker
    check_docker

    # Create deployment directory
    sudo mkdir -p $DEPLOY_DIR
    sudo chown $USER:$USER $DEPLOY_DIR

    log_info "Setup complete. Now run: ./deploy.sh deploy"
}

# Deploy the application
deploy() {
    log_info "Deploying AgentVerse to production..."

    # Check if .env exists
    if [ ! -f ".env" ]; then
        if [ -f ".env.production" ]; then
            log_warn ".env not found. Copying from .env.production..."
            cp .env.production .env
            log_warn "Please edit .env and update the OPENROUTER_API_KEY and SECRET_KEY!"
        else
            log_error ".env file not found. Please create it from .env.production"
            exit 1
        fi
    fi

    # Build and start containers
    log_info "Building Docker images..."
    docker compose -f $COMPOSE_FILE build --no-cache

    log_info "Starting containers..."
    docker compose -f $COMPOSE_FILE up -d

    log_info "Waiting for services to be healthy..."
    sleep 10

    # Show status
    docker compose -f $COMPOSE_FILE ps

    log_info "Deployment complete!"
    echo ""
    echo "========================================"
    echo "AgentVerse is now running!"
    echo "========================================"
    echo "Frontend: http://localhost:3003 (or your configured domain)"
    echo "API:      http://localhost:8001 (or your configured domain)"
    echo "API Docs: http://localhost:8001/docs"
    echo "========================================"
}

# Update the application
update() {
    log_info "Updating AgentVerse..."

    # Pull latest code (if using git)
    if [ -d ".git" ]; then
        log_info "Pulling latest code..."
        git pull
    fi

    # Rebuild and restart
    log_info "Rebuilding containers..."
    docker compose -f $COMPOSE_FILE build --no-cache

    log_info "Restarting services..."
    docker compose -f $COMPOSE_FILE up -d

    log_info "Update complete!"
}

# Show logs
logs() {
    SERVICE=${1:-""}
    if [ -z "$SERVICE" ]; then
        docker compose -f $COMPOSE_FILE logs -f --tail=100
    else
        docker compose -f $COMPOSE_FILE logs -f --tail=100 $SERVICE
    fi
}

# Stop all services
stop() {
    log_info "Stopping AgentVerse..."
    docker compose -f $COMPOSE_FILE down
    log_info "All services stopped"
}

# Restart all services
restart() {
    log_info "Restarting AgentVerse..."
    docker compose -f $COMPOSE_FILE restart
    log_info "All services restarted"
}

# Backup database
backup() {
    BACKUP_DIR="./backups"
    BACKUP_FILE="$BACKUP_DIR/agentverse_$(date +%Y%m%d_%H%M%S).sql"

    mkdir -p $BACKUP_DIR

    log_info "Creating database backup..."
    docker compose -f $COMPOSE_FILE exec -T postgres pg_dump -U agentverse agentverse > $BACKUP_FILE

    log_info "Backup saved to: $BACKUP_FILE"
}

# Show status
status() {
    docker compose -f $COMPOSE_FILE ps
}

# Show help
show_help() {
    echo "AgentVerse Deployment Script"
    echo ""
    echo "Usage: ./deploy.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup    - Initial server setup (install Docker)"
    echo "  deploy   - Build and deploy the application"
    echo "  update   - Pull latest code and redeploy"
    echo "  logs     - Show container logs (optional: service name)"
    echo "  stop     - Stop all services"
    echo "  restart  - Restart all services"
    echo "  backup   - Backup the database"
    echo "  status   - Show container status"
    echo "  help     - Show this help message"
}

# Main
case "${1:-help}" in
    setup)
        setup
        ;;
    deploy)
        deploy
        ;;
    update)
        update
        ;;
    logs)
        logs $2
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    backup)
        backup
        ;;
    status)
        status
        ;;
    help|*)
        show_help
        ;;
esac
