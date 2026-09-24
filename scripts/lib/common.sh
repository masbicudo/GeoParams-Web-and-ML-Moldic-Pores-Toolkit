#!/bin/sh
set -eu

: "${GEOPARAMS_SCRIPT_DIR:?Set GEOPARAMS_SCRIPT_DIR before loading common.sh}"

repo_dir=$(CDPATH='' cd -P "$GEOPARAMS_SCRIPT_DIR/.." && pwd)
app_dir="$repo_dir/geo_params_web"
docker_url="https://docs.docker.com/get-started/get-docker/"
dataset_url="https://drive.google.com/drive/folders/1s-NAWbgukQG-1Q3M5MpO808XRqA1QVw4?usp=sharing"
export repo_dir app_dir docker_url dataset_url

export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-geo-params-web}"
export GEO_PARAMS_IMAGE="${GEO_PARAMS_IMAGE:-geo-params-web-app:app-v2}"
export GEO_PARAMS_DATASET_IMAGE="${GEO_PARAMS_DATASET_IMAGE:-geo-params-datasets:app-v2}"

compose_style=

pause_for_user() {
    printf "Press Enter to continue: "
    read -r _answer || true
}

confirm() {
    prompt=$1
    printf "%s [y/N]: " "$prompt"
    read -r answer || answer=
    case "$answer" in
        y | Y | yes | YES) return 0 ;;
        *) return 1 ;;
    esac
}

retry_or_exit() {
    printf "Press Enter to check again, or type q to exit: "
    read -r answer || answer=q
    case "$answer" in
        q | Q) exit 1 ;;
    esac
}

detect_compose() {
    if docker compose version >/dev/null 2>&1; then
        compose_style=plugin
        return 0
    fi
    if command -v docker-compose >/dev/null 2>&1 &&
        docker-compose version >/dev/null 2>&1; then
        compose_style=standalone
        return 0
    fi
    return 1
}

ensure_docker() {
    while ! command -v docker >/dev/null 2>&1; do
        echo "Docker is required for this option."
        echo "Installation guide: $docker_url"
        retry_or_exit
    done
    while ! docker info >/dev/null 2>&1; do
        echo "Docker is installed, but its service is not running."
        echo "Start Docker Desktop or Docker Engine, then return here."
        retry_or_exit
    done
    while ! detect_compose; do
        echo "Docker Compose is not available."
        echo "Install the current Docker package from: $docker_url"
        retry_or_exit
    done
}

compose() {
    (
        cd "$app_dir"
        if [ "$compose_style" = plugin ]; then
            docker compose "$@"
        else
            docker-compose "$@"
        fi
    )
}

host_uid() {
    id -u 2>/dev/null || printf '0\n'
}

host_gid() {
    id -g 2>/dev/null || printf '0\n'
}

dataset_tool() {
    ensure_docker
    mkdir -p "$repo_dir/datasets"
    export GEO_PARAMS_UID="${GEO_PARAMS_UID:-$(host_uid)}"
    export GEO_PARAMS_GID="${GEO_PARAMS_GID:-$(host_gid)}"
    printf "Preparing the dataset manager... "
    if compose build dataset-manager >/dev/null; then
        echo "done"
    else
        echo "failed"
        return 1
    fi
    compose run --rm dataset-manager "$@"
}

command_status() {
    label=$1
    command_name=$2
    if command -v "$command_name" >/dev/null 2>&1; then
        printf "[OK]      %s\n" "$label"
        return 0
    fi
    printf "[MISSING] %s\n" "$label"
    return 1
}
