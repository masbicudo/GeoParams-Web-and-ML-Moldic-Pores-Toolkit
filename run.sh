#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
app_dir="$repo_dir/geo_params_web"
install_url="https://docs.docker.com/get-started/get-docker/"
dataset_url="https://drive.google.com/drive/folders/1s-NAWbgukQG-1Q3M5MpO808XRqA1QVw4?usp=sharing"
port="${GEO_PARAMS_PORT:-8181}"
image="${GEO_PARAMS_IMAGE:-geo-params-web-app:app-v2}"
log_file="$app_dir/log/docker-run.log"

export GEO_PARAMS_PORT="$port"
export GEO_PARAMS_IMAGE="$image"

retry_or_exit() {
    printf "Press Enter to check again, or type q to exit: "
    read -r answer || answer=q
    case "$answer" in
        q|Q) exit 1 ;;
    esac
}

run_step() {
    label=$1
    shift
    printf "%s... " "$label"
    if "$@" >>"$log_file" 2>&1; then
        echo "done"
    else
        echo "failed"
        echo "See geo_params_web/log/docker-run.log for details."
        exit 1
    fi
}

echo "GeoParams Web - guided startup"
echo

while ! command -v docker >/dev/null 2>&1; do
    echo "Docker is not installed or is not available in this terminal."
    echo "Installation guide: $install_url"
    echo "Install Docker, then return here. This script will check again."
    retry_or_exit
done

while ! docker compose version >/dev/null 2>&1; do
    echo "Docker Compose is not available."
    echo "Install the current Docker package from: $install_url"
    retry_or_exit
done

while ! docker info >/dev/null 2>&1; do
    echo "Docker is installed, but its service is not running."
    echo "Start Docker Desktop or Docker Engine, then return here."
    retry_or_exit
done

source_images="$repo_dir/datasets/article_thin_sections"
cache_dir="$app_dir/static/imgs_sections"
cache_ready=false
if [ -f "$cache_dir/metadata.json" ]; then
    for cached_image in "$cache_dir"/*.jpg "$cache_dir"/*.jpeg \
        "$cache_dir"/*.png; do
        if [ -f "$cached_image" ]; then
            cache_ready=true
            break
        fi
    done
fi

while [ ! -d "$source_images" ] && [ "$cache_ready" = false ]; do
    echo "The public petrographic image dataset was not found."
    echo "Download it from: $dataset_url"
    echo "Place it in: $repo_dir/datasets/article_thin_sections"
    retry_or_exit
done

case "$port" in
    ''|*[!0-9]*)
        echo "GEO_PARAMS_PORT must be a number between 1 and 65535."
        exit 1
        ;;
esac
if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
    echo "GEO_PARAMS_PORT must be a number between 1 and 65535."
    exit 1
fi

mkdir -p "$app_dir/static/output" "$app_dir/static/imgs_sections"
mkdir -p "$app_dir/data/uploads" "$app_dir/log"
: >"$log_file"

cd "$app_dir"
run_step "[1/4] Building the application" docker compose build
if [ -d "$source_images" ]; then
    run_step "[2/4] Preparing petrographic images" \
        docker compose run --rm prepare
else
    echo "[2/4] Using the existing petrographic image cache... done"
fi
run_step "[3/4] Starting the web application" docker compose up -d app nginx

printf "[4/4] Waiting for the web interface... "
attempt=0
until docker compose exec -T nginx \
    wget -qO- http://127.0.0.1/health >/dev/null 2>&1 \
    && docker compose exec -T nginx \
    wget -qO- http://127.0.0.1/geo-server/ >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 60 ]; then
        echo "failed"
        docker compose ps >>"$log_file" 2>&1 || true
        docker compose logs --no-color --tail=200 >>"$log_file" 2>&1 || true
        echo "The application did not become ready."
        echo "See geo_params_web/log/docker-run.log for details."
        exit 1
    fi
    sleep 2
done
echo "done"

url="http://localhost:${port}/geo-server/"
echo
echo "GeoParams Web is ready: $url"
echo "Uploads and results remain in geo_params_web/data/uploads/."
echo
echo "What would you like to do now?"
echo "  1) Keep the application running in the background"
echo "  2) Stop the application now"
printf "Choice [1]: "
read -r choice || choice=1

case "$choice" in
    2)
        run_step "Stopping the application" docker compose stop
        echo "The application is stopped. Its saved data was preserved."
        ;;
    *)
        echo "The application will keep running in the background."
        ;;
esac

echo "Use ./remove-docker-app.sh to stop or remove its Docker resources."
