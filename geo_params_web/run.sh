#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

port="${GEO_PARAMS_PORT:-8181}"
health_url="http://127.0.0.1:${port}"
display_url="http://localhost:${port}"
started=false
wait_pid=""
log_file="$(pwd)/log/docker-run.log"
log_display="log/docker-run.log"

cleanup() {
    status=$?
    trap - EXIT HUP INT TERM
    if [ -n "$wait_pid" ]; then
        kill "$wait_pid" 2>/dev/null || true
    fi
    if [ "$started" = true ]; then
        echo
        printf "Stopping GeoParams Web... "
        if docker compose down >>"$log_file" 2>&1; then
            echo "done"
        else
            echo "failed"
            echo "Details: $log_display" >&2
        fi
    fi
    exit "$status"
}

trap cleanup EXIT HUP INT TERM

run_step() {
    label=$1
    shift

    printf "%s... " "$label"
    if "$@" >>"$log_file" 2>&1; then
        echo "done"
    else
        echo "failed"
        echo "Details: $log_display" >&2
        exit 1
    fi
}

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker was not found. Install Docker Desktop or Docker Engine first." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "[ERROR] Docker Compose is not available through 'docker compose'." >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] Docker is installed, but the Docker daemon is not running." >&2
    exit 1
fi

if [ ! -d "../datasets/article_thin_sections" ]; then
    echo "[ERROR] The article thin-section dataset was not found." >&2
    echo "Expected directory: $(cd .. && pwd)/datasets/article_thin_sections" >&2
    echo "Download the public dataset into the repository-level datasets directory." >&2
    exit 1
fi

mkdir -p static/output static/imgs_sections log
: >"$log_file"

run_step "[1/4] Building the application image" docker compose build

run_step "[2/4] Preparing the local image cache" docker compose run --rm prepare

run_step "[3/4] Starting the applications" docker compose up -d app nginx
started=true

printf "[4/4] Waiting for the web interface... "
attempt=0
until curl -fsS "${health_url}/health" >/dev/null 2>&1 \
    && curl -fsS "${health_url}/geo-server/" >/dev/null 2>&1 \
    && curl -fsS "${health_url}/geo-server/stats/_stcore/health" >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 60 ]; then
        echo "failed"
        docker compose ps >>"$log_file" 2>&1 || true
        docker compose logs --no-color --tail=200 >>"$log_file" 2>&1 || true
        echo "The web interface did not become ready." >&2
        echo "Details: $log_display" >&2
        exit 1
    fi
    sleep 2
done
echo "done"

echo
echo "[OK] GeoParams Web is running."
echo "Data collection: ${display_url}/geo-server/"
echo "Statistics:      ${display_url}/geo-server/stats/"
echo
echo "Keep this terminal open while using the applications."
echo "Press Ctrl+C or close the terminal to stop the servers."
echo

while :; do
    sleep 3600 &
    wait_pid=$!
    wait "$wait_pid"
    wait_pid=""
done
