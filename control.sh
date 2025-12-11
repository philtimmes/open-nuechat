#!/bin/bash

# =============================================================================
# Open-NueChat - Docker Controller
# =============================================================================
# Usage: ./control.sh [command] [options]
#
# Commands:
#   build       Build Docker images
#   start       Build and start containers
#   stop        Stop running containers
#   down        Stop and remove containers
#   restart     Restart containers
#   logs        View container logs
#   status      Show container status
#   shell       Open shell in container
#   test        Run tests
#   clean       Remove containers, volumes, and images
#   dev         Start in development mode
#   help        Show this help message
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_NAME="open-nuechat"
COMPOSE_FILE="docker-compose.yml"
COMPOSE_DEV_FILE="docker-compose.dev.yml"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                   Open-NueChat Docker Controller                   ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_help() {
    echo -e "${BLUE}Usage:${NC} ./control.sh [command] [options]"
    echo ""
    echo -e "${BLUE}Commands:${NC}"
    echo -e "  ${GREEN}build${NC}              Build Docker images"
    echo -e "    --no-cache       Build without using cache"
    echo -e "    --profile NAME   Build specific profile (rocm, cuda, cpu)"
    echo ""
    echo -e "  ${GREEN}start${NC}              Build and start containers"
    echo -e "    -d, --detach     Run in detached mode"
    echo -e "    --build          Force rebuild before starting"
    echo -e "    --profile NAME   Use specific profile (rocm, cuda, cpu)"
    echo ""
    echo -e "  ${GREEN}stop${NC}               Stop running containers"
    echo ""
    echo -e "  ${GREEN}down${NC}               Stop and remove containers"
    echo -e "    -v, --volumes    Also remove volumes"
    echo ""
    echo -e "  ${GREEN}restart${NC}            Restart containers"
    echo ""
    echo -e "  ${GREEN}logs${NC}               View container logs"
    echo -e "    -f, --follow     Follow log output"
    echo -e "    -n, --tail N     Number of lines to show (default: 100)"
    echo ""
    echo -e "  ${GREEN}status${NC}             Show container status"
    echo ""
    echo -e "  ${GREEN}shell${NC}              Open shell in container"
    echo ""
    echo -e "  ${GREEN}test${NC}               Run tests"
    echo -e "    --coverage       Include coverage report"
    echo ""
    echo -e "  ${GREEN}clean${NC}              Clean up Docker resources"
    echo -e "    --all            Remove images, volumes, and orphans"
    echo -e "    --volumes        Remove only volumes"
    echo -e "    --images         Remove only images"
    echo ""
    echo -e "  ${GREEN}db${NC}                 Database operations"
    echo -e "    migrate          Run database migrations"
    echo -e "    seed             Seed database with sample data"
    echo -e "    reset            Reset database (DESTRUCTIVE)"
    echo ""
    echo -e "  ${GREEN}tts${NC}                TTS service control"
    echo -e "    start            Start TTS service"
    echo -e "    stop             Stop TTS service"
    echo -e "    status           Show TTS service status"
    echo ""
    echo -e "  ${GREEN}image${NC}              Image generation service control"
    echo -e "    start            Start image generation service"
    echo -e "    stop             Stop image generation service"
    echo -e "    status           Show image service status"
    echo -e "    logs             View image service logs"
    echo ""
    echo -e "  ${GREEN}migrate${NC}            Migrate data from Docker volumes to /opt/nuechat/_data"
    echo -e "    --dry-run        Show what would be copied without copying"
    echo ""
    echo -e "  ${GREEN}faiss${NC}              Check FAISS wheel status"
    echo ""
    echo -e "  ${GREEN}faiss-build${NC}        Build FAISS wheel for specific profile"
    echo -e "    --profile NAME   Target profile (rocm, cuda, cpu)"
    echo -e "    --force          Rebuild even if wheel exists"
    echo ""
    echo -e "  ${GREEN}help${NC}               Show this help message"
    echo ""
    echo -e "${BLUE}Profiles:${NC}"
    echo -e "  ${CYAN}rocm${NC}               AMD GPU with ROCm (requires ROCm 6.0+)"
    echo -e "  ${CYAN}cuda${NC}               NVIDIA GPU with CUDA (requires nvidia-container-toolkit)"
    echo -e "  ${CYAN}cpu${NC}                CPU only (no GPU required, slower)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  ./control.sh build --profile cuda       # Build for NVIDIA GPU"
    echo "  ./control.sh build --profile cpu        # Build for CPU only"
    echo "  ./control.sh start -d --profile rocm    # Start with AMD GPU"
    echo "  ./control.sh start -d --profile cuda    # Start with NVIDIA GPU"
    echo "  ./control.sh start -d --profile cpu     # Start CPU-only"
    echo "  ./control.sh faiss-build --profile rocm # Build FAISS for ROCm"
    echo "  ./control.sh logs -f                    # Follow logs"
    echo "  ./control.sh shell                      # Shell into container"
    echo "  ./control.sh build --no-cache --profile rocm  # Full rebuild"
    echo "  ./control.sh clean --all                # Full cleanup"
    echo ""
    echo -e "${BLUE}First-time Setup (ROCm):${NC}"
    echo "  1. Copy .env.example to .env and configure"
    echo "  2. ./control.sh faiss-build --profile rocm  # ~12 min, once only"
    echo "  3. ./control.sh build --profile rocm"
    echo "  4. ./control.sh start -d --profile rocm"
    echo ""
    echo -e "${BLUE}First-time Setup (CUDA/CPU):${NC}"
    echo "  1. Copy .env.example to .env and configure"
    echo "  2. ./control.sh build --profile <cuda|cpu>"
    echo "  3. ./control.sh start -d --profile <cuda|cpu>"
    echo ""
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
}

check_compose() {
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        log_error "Docker Compose is not installed. Please install Docker Compose."
        exit 1
    fi
}

# Check if .env file exists, create from example if not
check_env() {
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            log_warning ".env file not found. Creating from .env.example"
            cp .env.example .env
            log_info "Please edit .env file with your configuration"
        else
            log_warning ".env file not found. Some features may not work correctly."
        fi
    fi
}

# -----------------------------------------------------------------------------
# Command Functions
# -----------------------------------------------------------------------------

cmd_build() {
    local no_cache=""
    local profile="rocm"  # Default profile
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-cache)
                no_cache="--no-cache"
                shift
                ;;
            --profile)
                profile="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Validate profile
    case $profile in
        rocm|cuda|cpu)
            ;;
        *)
            log_error "Invalid profile: $profile (must be rocm, cuda, or cpu)"
            exit 1
            ;;
    esac
    
    log_info "Building Docker images with profile: $profile"
    $COMPOSE_CMD --profile $profile build $no_cache
    
    log_success "Build completed successfully!"
}

# Check FAISS wheel status
cmd_faiss_status() {
    echo -e "${BLUE}FAISS Wheel Status:${NC}"
    echo ""
    
    echo -e "${CYAN}ROCm wheel:${NC}"
    if ls backend/wheels/faiss*rocm*.whl 1>/dev/null 2>&1; then
        log_success "Found:"
        ls -la backend/wheels/faiss*rocm*.whl
    elif ls backend/wheels/faiss*.whl 1>/dev/null 2>&1; then
        log_success "Found (generic):"
        ls -la backend/wheels/faiss*.whl
    else
        log_warning "Not found - run: ./control.sh faiss-build --profile rocm"
    fi
    echo ""
    
    echo -e "${CYAN}CUDA:${NC}"
    echo "  No wheel needed - uses pip install faiss-gpu"
    echo ""
    
    echo -e "${CYAN}CPU:${NC}"
    echo "  No wheel needed - uses pip install faiss-cpu"
}

# Build FAISS wheel for specific profile
cmd_faiss_build() {
    local profile=""
    local force=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --profile)
                profile="$2"
                shift 2
                ;;
            --force)
                force="true"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: ./control.sh faiss-build --profile <rocm|cuda|cpu> [--force]"
                exit 1
                ;;
        esac
    done
    
    if [ -z "$profile" ]; then
        log_error "Profile required"
        echo "Usage: ./control.sh faiss-build --profile <rocm|cuda|cpu> [--force]"
        exit 1
    fi
    
    case $profile in
        rocm)
            cmd_faiss_build_rocm "$force"
            ;;
        cuda)
            echo ""
            log_info "CUDA profile does not require a pre-built wheel."
            echo ""
            echo "The Dockerfile.cuda uses:"
            echo "  pip install faiss-gpu"
            echo ""
            echo "This installs the official NVIDIA CUDA-enabled FAISS from PyPI."
            log_success "No action needed for CUDA profile."
            ;;
        cpu)
            echo ""
            log_info "CPU profile does not require a pre-built wheel."
            echo ""
            echo "The Dockerfile.cpu uses:"
            echo "  pip install faiss-cpu"
            echo ""
            echo "This installs the official CPU-only FAISS from PyPI."
            log_success "No action needed for CPU profile."
            ;;
        *)
            log_error "Invalid profile: $profile (must be rocm, cuda, or cpu)"
            exit 1
            ;;
    esac
}

# Build FAISS wheel for ROCm
cmd_faiss_build_rocm() {
    local force="$1"
    local wheels_dir="backend/wheels"
    
    # Check if wheel already exists
    if ls "$wheels_dir"/faiss*.whl 1>/dev/null 2>&1; then
        if [ "$force" != "true" ]; then
            echo ""
            log_success "FAISS ROCm wheel already exists:"
            ls -la "$wheels_dir"/faiss*.whl
            echo ""
            echo "FAISS will NOT be rebuilt."
            echo "Use --force to rebuild anyway."
            return 0
        else
            log_warning "Force rebuild requested. Removing existing wheel..."
            rm -f "$wheels_dir"/faiss*.whl
        fi
    fi
    
    # Extract base image from Dockerfile
    local dockerfile="backend/Dockerfile"
    local base_image=$(grep -E "^FROM.*rocm" "$dockerfile" | head -1 | awk '{print $2}')
    
    if [ -z "$base_image" ]; then
        log_error "Could not extract ROCm base image from $dockerfile"
        exit 1
    fi
    
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                   Building FAISS ROCm Wheel                        ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    log_info "Base image: $base_image"
    log_info "This takes ~12 minutes but only happens ONCE."
    echo ""
    
    mkdir -p "$wheels_dir"
    
    # Build in a temporary container
    docker run --rm \
        -v "$(pwd)/$wheels_dir:/output" \
        "$base_image" \
        bash -c '
            set -e
            echo "Python version: $(python --version)"
            
            echo "Installing build dependencies..."
            apt-get update && apt-get install -y build-essential cmake git libopenblas-dev swig
            
            echo "Installing faiss-cpu for dependencies..."
            pip install faiss-cpu && pip uninstall -y faiss-cpu
            
            echo "Cloning faiss-wheels..."
            git clone --recursive https://github.com/faiss-wheels/faiss-wheels.git /tmp/faiss-wheels
            cd /tmp/faiss-wheels
            
            echo "Building FAISS with ROCm support..."
            pip install pipx
            FAISS_GPU_SUPPORT=ROCM FAISS_OPT_LEVELS=generic pipx run build --wheel
            
            echo "Copying wheel to output..."
            cp dist/faiss*.whl /output/
            
            echo "Done!"
        '
    
    # Verify wheel was created
    if ls "$wheels_dir"/faiss*.whl 1>/dev/null 2>&1; then
        echo ""
        log_success "FAISS ROCm wheel built successfully!"
        ls -la "$wheels_dir"/faiss*.whl
        echo ""
        echo "This wheel will be used for all ROCm builds."
    else
        log_error "Wheel file not created!"
        exit 1
    fi
}

cmd_start() {
    local detach=""
    local build=""
    local profile="rocm"  # Default profile
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--detach)
                detach="-d"
                shift
                ;;
            --build)
                build="--build"
                shift
                ;;
            --profile)
                profile="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Validate profile
    case $profile in
        rocm|cuda|cpu)
            ;;
        *)
            log_error "Invalid profile: $profile (must be rocm, cuda, or cpu)"
            exit 1
            ;;
    esac
    
    check_env
    
    log_info "Starting Open-NueChat with profile: $profile"
    $COMPOSE_CMD --profile $profile up $detach $build
    
    if [ -n "$detach" ]; then
        log_success "Open-NueChat started in background"
        echo ""
        echo "View logs with: ./control.sh logs -f"
        echo "Stop with:      ./control.sh stop"
    fi
}

cmd_stop() {
    log_info "Stopping containers..."
    $COMPOSE_CMD stop
    log_success "Containers stopped"
}

cmd_down() {
    local volumes=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--volumes)
                volumes="-v"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    log_info "Stopping and removing containers..."
    $COMPOSE_CMD down $volumes
    log_success "Containers removed"
}

cmd_restart() {
    log_info "Restarting container..."
    $COMPOSE_CMD restart
    log_success "Container restarted"
}

cmd_logs() {
    local follow=""
    local tail="100"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--follow)
                follow="-f"
                shift
                ;;
            -n|--tail)
                tail="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    $COMPOSE_CMD logs $follow --tail=$tail
}

cmd_status() {
    echo -e "${BLUE}Container Status:${NC}"
    echo ""
    $COMPOSE_CMD ps
    echo ""
    
    # Check health status
    echo -e "${BLUE}Health Status:${NC}"
    echo ""
    
    # Backend health
    if docker ps --filter "name=${PROJECT_NAME}-backend" --filter "status=running" -q | grep -q .; then
        local health=$(docker inspect --format='{{.State.Health.Status}}' ${PROJECT_NAME}-backend 2>/dev/null || echo "unknown")
        if [ "$health" == "healthy" ]; then
            echo -e "  Backend:  ${GREEN}●${NC} Healthy"
        elif [ "$health" == "unhealthy" ]; then
            echo -e "  Backend:  ${RED}●${NC} Unhealthy"
        else
            echo -e "  Backend:  ${YELLOW}●${NC} $health"
        fi
    else
        echo -e "  Backend:  ${RED}●${NC} Not running"
    fi
    
    # Frontend health
    if docker ps --filter "name=${PROJECT_NAME}-frontend" --filter "status=running" -q | grep -q .; then
        echo -e "  Frontend: ${GREEN}●${NC} Running"
    else
        echo -e "  Frontend: ${RED}●${NC} Not running"
    fi
    
    echo ""
}

cmd_shell() {
    log_info "Opening shell in container..."
    $COMPOSE_CMD exec open-nuechat /bin/bash
}

cmd_test() {
    local coverage=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --coverage)
                coverage="--cov=app --cov-report=html --cov-report=term"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    log_info "Running tests..."
    if $COMPOSE_CMD exec -T open-nuechat pytest $coverage tests/ -v; then
        log_success "All tests passed!"
    else
        log_error "Some tests failed"
        return 1
    fi
}

cmd_clean() {
    local all=false
    local volumes=false
    local images=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                all=true
                shift
                ;;
            --volumes)
                volumes=true
                shift
                ;;
            --images)
                images=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    log_warning "This will remove Docker resources. Continue? [y/N]"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Cancelled"
        exit 0
    fi
    
    log_info "Stopping containers..."
    $COMPOSE_CMD down
    
    if [ "$all" = true ] || [ "$volumes" = true ]; then
        log_info "Removing volumes..."
        $COMPOSE_CMD down -v
        docker volume ls --filter "name=${PROJECT_NAME}" -q | xargs -r docker volume rm 2>/dev/null || true
    fi
    
    if [ "$all" = true ] || [ "$images" = true ]; then
        log_info "Removing images..."
        docker images --filter "reference=${PROJECT_NAME}*" -q | xargs -r docker rmi -f 2>/dev/null || true
    fi
    
    if [ "$all" = true ]; then
        log_info "Removing orphan containers and networks..."
        docker system prune -f --filter "label=com.docker.compose.project=${PROJECT_NAME}" 2>/dev/null || true
    fi
    
    log_success "Cleanup completed"
}

cmd_dev() {
    local service=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backend)
                service="backend"
                shift
                ;;
            --frontend)
                service="frontend"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    check_env
    
    # Check if dev compose file exists
    if [ ! -f "$COMPOSE_DEV_FILE" ]; then
        log_info "Creating development compose file..."
        create_dev_compose
    fi
    
    log_info "Starting in development mode..."
    
    if [ -n "$service" ]; then
        $COMPOSE_CMD -f $COMPOSE_FILE -f $COMPOSE_DEV_FILE up $service
    else
        $COMPOSE_CMD -f $COMPOSE_FILE -f $COMPOSE_DEV_FILE up
    fi
}

cmd_db() {
    local action="$1"
    shift
    
    case $action in
        migrate)
            log_info "Running database migrations..."
            $COMPOSE_CMD exec open-nuechat python -c "
from app.db.database import engine
from app.models.models import Base
import asyncio

async def migrate():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print('Migrations completed')

asyncio.run(migrate())
"
            log_success "Migrations completed"
            ;;
        seed)
            log_info "Seeding database..."
            $COMPOSE_CMD exec open-nuechat python -c "
from app.db.seed import seed_database
import asyncio
asyncio.run(seed_database())
print('Database seeded')
"
            log_success "Database seeded"
            ;;
        reset)
            log_warning "This will DELETE all data. Are you sure? [y/N]"
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                log_info "Resetting database..."
                $COMPOSE_CMD exec open-nuechat rm -f /app/data/nuechat.db
                $COMPOSE_CMD restart open-nuechat
                sleep 5
                cmd_db migrate
                log_success "Database reset completed"
            else
                log_info "Cancelled"
            fi
            ;;
        *)
            log_error "Unknown database action: $action"
            log_info "Available actions: migrate, seed, reset"
            exit 1
            ;;
    esac
}

cmd_tts() {
    local action="${1:-status}"
    shift 2>/dev/null || true
    
    case $action in
        cpu)
            # Stop any existing TTS service first
            log_info "Stopping any existing TTS services..."
            $COMPOSE_CMD --profile cpu-tts stop tts-service 2>/dev/null || true
            $COMPOSE_CMD --profile rocm-tts stop tts-service-rocm 2>/dev/null || true
            
            log_info "Starting CPU-only TTS service..."
            $COMPOSE_CMD --profile cpu-tts up -d tts-service
            log_success "CPU TTS service started"
            echo ""
            echo -e "  ${CYAN}TTS API:${NC}  http://localhost:${TTS_PORT:-8033}"
            echo -e "  ${CYAN}Health:${NC}   http://localhost:${TTS_PORT:-8033}/health"
            echo ""
            ;;
        rocm|gpu)
            # Stop any existing TTS service first
            log_info "Stopping any existing TTS services..."
            $COMPOSE_CMD --profile cpu-tts stop tts-service 2>/dev/null || true
            $COMPOSE_CMD --profile rocm-tts stop tts-service-rocm 2>/dev/null || true
            
            log_info "Starting ROCm GPU TTS service..."
            $COMPOSE_CMD --profile rocm-tts up -d tts-service-rocm
            log_success "ROCm TTS service started"
            echo ""
            echo -e "  ${CYAN}TTS API:${NC}  http://localhost:${TTS_PORT:-8033}"
            echo -e "  ${CYAN}Health:${NC}   http://localhost:${TTS_PORT:-8033}/health"
            echo ""
            log_info "Checking GPU detection..."
            sleep 3
            curl -s http://localhost:${TTS_PORT:-8033}/health 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('gpu'):
        print(f\"  GPU: {data['gpu'].get('name', 'Unknown')}\")
        print(f\"  VRAM: {data['gpu'].get('vram_total_gb', 0):.1f} GB\")
        if data['gpu'].get('rocm_version'):
            print(f\"  ROCm: {data['gpu']['rocm_version']}\")
    else:
        print('  GPU: Not detected (running on CPU)')
except:
    print('  (waiting for service to start...)')
" || true
            echo ""
            ;;
        stop)
            log_info "Stopping TTS services..."
            $COMPOSE_CMD --profile cpu-tts stop tts-service 2>/dev/null || true
            $COMPOSE_CMD --profile rocm-tts stop tts-service-rocm 2>/dev/null || true
            log_success "TTS services stopped"
            ;;
        status)
            echo -e "${BLUE}TTS Service Status:${NC}"
            echo ""
            
            # Check CPU service
            if docker ps --filter "name=nexus-tts" --filter "status=running" -q 2>/dev/null | grep -q .; then
                local container=$(docker ps --filter "name=nexus-tts" --format "{{.Names}}" 2>/dev/null | head -1)
                if [[ "$container" == *"rocm"* ]]; then
                    echo -e "  ROCm TTS:  ${GREEN}●${NC} Running"
                else
                    echo -e "  CPU TTS:   ${GREEN}●${NC} Running"
                fi
                
                # Get health info
                curl -s http://localhost:${TTS_PORT:-8033}/health 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Device:    {data.get('device', 'unknown')}\")
    print(f\"  Queue:     {data.get('queue_size', 0)}/{data.get('max_queue', 100)}\")
    if data.get('gpu'):
        print(f\"  GPU:       {data['gpu'].get('name', 'Unknown')}\")
        print(f\"  VRAM Used: {data['gpu'].get('vram_used_gb', 0):.2f} GB\")
except:
    pass
" 2>/dev/null || true
            else
                echo -e "  TTS:       ${RED}●${NC} Not running"
            fi
            echo ""
            ;;
        logs)
            if docker ps --filter "name=nexus-tts" --filter "status=running" -q 2>/dev/null | grep -q .; then
                local container=$(docker ps --filter "name=nexus-tts" --format "{{.Names}}" 2>/dev/null | head -1)
                docker logs -f "$container"
            else
                log_error "No TTS service is running"
            fi
            ;;
        *)
            log_error "Unknown TTS action: $action"
            log_info "Available actions: cpu, rocm, stop, status, logs"
            exit 1
            ;;
    esac
}

cmd_image() {
    local action="${1:-status}"
    shift 2>/dev/null || true
    
    case $action in
        start|rocm)
            log_info "Starting Image Generation service..."
            $COMPOSE_CMD --profile image-gen up -d image-service
            log_success "Image Generation service started"
            echo ""
            echo -e "  ${CYAN}API:${NC}     http://localhost:${IMAGE_GEN_PORT:-8034}"
            echo -e "  ${CYAN}Health:${NC}  http://localhost:${IMAGE_GEN_PORT:-8034}/health"
            echo ""
            log_info "Checking GPU detection (model loading may take a while)..."
            sleep 5
            curl -s http://localhost:${IMAGE_GEN_PORT:-8034}/health 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Status: {data.get('status', 'unknown')}\")
    print(f\"  Model: {data.get('model', 'unknown')}\")
    if data.get('gpu'):
        print(f\"  GPU: {data['gpu'].get('name', 'Unknown')}\")
        print(f\"  VRAM: {data['gpu'].get('vram_total_gb', 0):.1f} GB\")
except:
    print('  (waiting for service to start - model download may take several minutes...)')
" || true
            echo ""
            ;;
        stop)
            log_info "Stopping Image Generation service..."
            $COMPOSE_CMD --profile image-gen stop image-service 2>/dev/null || true
            log_success "Image Generation service stopped"
            ;;
        status)
            echo -e "${BLUE}Image Generation Service Status:${NC}"
            echo ""
            
            if docker ps --filter "name=nexus-image-gen" --filter "status=running" -q 2>/dev/null | grep -q .; then
                echo -e "  Image Gen:  ${GREEN}●${NC} Running"
                
                curl -s http://localhost:${IMAGE_GEN_PORT:-8034}/health 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Model:     {data.get('model', 'unknown')}\")
    print(f\"  Device:    {data.get('device', 'unknown')}\")
    print(f\"  Queue:     {data.get('queue_size', 0)}/{data.get('max_queue', 20)}\")
    if data.get('gpu'):
        print(f\"  GPU:       {data['gpu'].get('name', 'Unknown')}\")
        print(f\"  VRAM Used: {data['gpu'].get('vram_used_gb', 0):.2f} GB\")
except:
    pass
" 2>/dev/null || true
            else
                echo -e "  Image Gen: ${RED}●${NC} Not running"
            fi
            echo ""
            ;;
        logs)
            if docker ps --filter "name=nexus-image-gen" --filter "status=running" -q 2>/dev/null | grep -q .; then
                docker logs -f nexus-image-gen
            else
                log_error "Image Generation service is not running"
            fi
            ;;
        *)
            log_error "Unknown image action: $action"
            log_info "Available actions: start, stop, status, logs"
            exit 1
            ;;
    esac
}

cmd_migrate() {
    local dry_run=false
    local target_dir="/opt/nuechat/_data"
    local docker_volumes="/var/lib/docker/volumes"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    echo -e "${BLUE}Open-NueChat Data Migration${NC}"
    echo ""
    echo "This will migrate data from Docker volumes to: $target_dir"
    echo ""
    
    # Define volume mappings (old volume name -> subdirectory in target)
    declare -A volume_mappings=(
        ["nuechat_app-data"]="."           # Contains nuechat.db -> /opt/nuechat/_data/
        ["nuechat_uploads-data"]="."       # Contains generated/ -> /opt/nuechat/_data/
        ["nuechat_faiss-indexes"]="faiss_indexes"
        ["nuechat_image-gen-cache"]="hf_cache"
    )
    
    # Check what exists
    echo -e "${CYAN}Checking existing Docker volumes...${NC}"
    local found_any=false
    
    for volume in "${!volume_mappings[@]}"; do
        local volume_path="$docker_volumes/$volume/_data"
        if [ -d "$volume_path" ]; then
            local size=$(du -sh "$volume_path" 2>/dev/null | cut -f1)
            echo -e "  ${GREEN}✓${NC} $volume ($size)"
            found_any=true
        else
            echo -e "  ${YELLOW}○${NC} $volume (not found)"
        fi
    done
    echo ""
    
    if [ "$found_any" = false ]; then
        log_warning "No existing Docker volumes found. Nothing to migrate."
        echo ""
        echo "If this is a fresh install, just create the target directory:"
        echo "  sudo mkdir -p $target_dir"
        echo "  sudo chown -R \$USER:\$USER /opt/nuechat"
        exit 0
    fi
    
    if [ "$dry_run" = true ]; then
        echo -e "${YELLOW}DRY RUN - No files will be copied${NC}"
        echo ""
    fi
    
    # Check if target exists
    if [ -d "$target_dir" ] && [ "$(ls -A $target_dir 2>/dev/null)" ]; then
        log_warning "Target directory $target_dir already contains files!"
        echo ""
        ls -la "$target_dir"
        echo ""
        if [ "$dry_run" = false ]; then
            read -p "Continue and merge? Files will NOT be overwritten. (y/N) " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Migration cancelled."
                exit 0
            fi
        fi
    fi
    
    # Create target directory
    if [ "$dry_run" = false ]; then
        log_info "Creating target directory..."
        sudo mkdir -p "$target_dir"
        sudo mkdir -p "$target_dir/faiss_indexes"
        sudo mkdir -p "$target_dir/hf_cache"
        sudo mkdir -p "$target_dir/generated"
    else
        echo "Would create: $target_dir"
        echo "Would create: $target_dir/faiss_indexes"
        echo "Would create: $target_dir/hf_cache"
        echo "Would create: $target_dir/generated"
    fi
    echo ""
    
    # Copy each volume
    echo -e "${CYAN}Copying data...${NC}"
    for volume in "${!volume_mappings[@]}"; do
        local volume_path="$docker_volumes/$volume/_data"
        local target_subdir="${volume_mappings[$volume]}"
        local full_target="$target_dir"
        
        if [ "$target_subdir" != "." ]; then
            full_target="$target_dir/$target_subdir"
        fi
        
        if [ -d "$volume_path" ]; then
            echo -e "  Copying $volume -> $full_target"
            
            if [ "$dry_run" = false ]; then
                # Use rsync to copy without overwriting existing files
                sudo rsync -av --ignore-existing "$volume_path/" "$full_target/"
            else
                echo "    Would copy: $volume_path/ -> $full_target/"
                ls -la "$volume_path/" 2>/dev/null | head -10
                echo ""
            fi
        fi
    done
    echo ""
    
    # Set ownership
    if [ "$dry_run" = false ]; then
        log_info "Setting ownership..."
        sudo chown -R $USER:$USER /opt/nuechat
    else
        echo "Would set ownership: chown -R $USER:$USER /opt/nuechat"
    fi
    echo ""
    
    # Show result
    if [ "$dry_run" = false ]; then
        log_success "Migration complete!"
        echo ""
        echo -e "${CYAN}New data location:${NC}"
        echo "  Database:    $target_dir/nuechat.db"
        echo "  Images:      $target_dir/generated/"
        echo "  FAISS:       $target_dir/faiss_indexes/"
        echo "  HF Cache:    $target_dir/hf_cache/"
        echo ""
        
        if [ -f "$target_dir/nuechat.db" ]; then
            local db_size=$(du -h "$target_dir/nuechat.db" | cut -f1)
            echo -e "  ${GREEN}✓${NC} Database: $db_size"
        fi
        
        if [ -d "$target_dir/generated" ]; then
            local img_count=$(find "$target_dir/generated" -name "*.png" 2>/dev/null | wc -l)
            echo -e "  ${GREEN}✓${NC} Generated images: $img_count files"
        fi
        echo ""
        
        log_info "You can now start Open-NueChat with: ./control.sh start -d"
        echo ""
        log_warning "After verifying everything works, you can remove old Docker volumes with:"
        echo "  docker volume rm nuechat_app-data nuechat_uploads-data nuechat_faiss-indexes nuechat_image-gen-cache"
    else
        echo -e "${YELLOW}Dry run complete. Run without --dry-run to perform migration.${NC}"
    fi
}

create_dev_compose() {
    cat > "$COMPOSE_DEV_FILE" << 'EOF'
# Development overrides for docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    volumes:
      - ./backend/app:/app/app:ro
      - backend-data:/app/data
    environment:
      - DEBUG=true
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    volumes:
      - ./frontend/src:/app/src:ro
      - ./frontend/public:/app/public:ro
    ports:
      - "5173:5173"
    command: npm run dev -- --host 0.0.0.0
EOF
    
    # Create frontend dev Dockerfile if it doesn't exist
    if [ ! -f "frontend/Dockerfile.dev" ]; then
        cat > "frontend/Dockerfile.dev" << 'EOF'
FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
EOF
    fi
    
    log_success "Development compose file created"
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

main() {
    # Change to script directory
    cd "$(dirname "$0")"
    
    # Check prerequisites
    check_docker
    check_compose
    
    # Parse command
    local command="${1:-help}"
    shift 2>/dev/null || true
    
    case $command in
        build)
            print_banner
            cmd_build "$@"
            ;;
        start|up)
            print_banner
            cmd_start "$@"
            ;;
        stop)
            print_banner
            cmd_stop "$@"
            ;;
        down)
            print_banner
            cmd_down "$@"
            ;;
        restart)
            print_banner
            cmd_restart "$@"
            ;;
        logs)
            cmd_logs "$@"
            ;;
        status|ps)
            print_banner
            cmd_status "$@"
            ;;
        shell|exec|sh)
            cmd_shell "$@"
            ;;
        test)
            print_banner
            cmd_test "$@"
            ;;
        clean)
            print_banner
            cmd_clean "$@"
            ;;
        dev)
            print_banner
            cmd_dev "$@"
            ;;
        db)
            print_banner
            cmd_db "$@"
            ;;
        tts)
            print_banner
            cmd_tts "$@"
            ;;
        image|imagegen)
            print_banner
            cmd_image "$@"
            ;;
        migrate)
            print_banner
            cmd_migrate "$@"
            ;;
        faiss)
            print_banner
            cmd_faiss_status
            ;;
        faiss-build)
            print_banner
            cmd_faiss_build "$@"
            ;;
        help|--help|-h)
            print_banner
            print_help
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            print_help
            exit 1
            ;;
    esac
}

main "$@"
