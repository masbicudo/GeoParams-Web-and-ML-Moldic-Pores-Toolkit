#!/bin/sh
set -eu

GEOPARAMS_SCRIPT_DIR=$(CDPATH='' cd -P "$(dirname "$0")" && pwd)
export GEOPARAMS_SCRIPT_DIR
# shellcheck source=scripts/lib/common.sh
. "$GEOPARAMS_SCRIPT_DIR/lib/common.sh"

port="${GEO_PARAMS_PORT:-8181}"
log_file="$app_dir/log/docker-run.log"
export GEO_PARAMS_PORT="$port"

run_step() {
    label=$1
    shift
    printf "%s... " "$label"
    if "$@" >>"$log_file" 2>&1; then
        echo "done"
    else
        echo "failed"
        echo "See geo_params_web/log/docker-run.log for details."
        return 1
    fi
}

echo "GeoParams Web - guided startup"
echo
ensure_docker

case "$port" in
    '' | *[!0-9]*)
        echo "GEO_PARAMS_PORT must be a number between 1 and 65535."
        exit 1
        ;;
esac
if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
    echo "GEO_PARAMS_PORT must be a number between 1 and 65535."
    exit 1
fi

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
    echo "The application image dataset is not available yet."
    echo "Use Dataset management in GeoParams to download it."
    echo "Manual download: $dataset_url"
    retry_or_exit
done

mkdir -p "$app_dir/static/output" "$app_dir/static/imgs_sections"
mkdir -p "$app_dir/data/uploads" "$app_dir/log"
: >"$log_file"

run_step "[1/4] Building the application" compose build app
if [ -d "$source_images" ]; then
    run_step "[2/4] Preparing petrographic images" \
        compose run --rm prepare
else
    echo "[2/4] Using the existing petrographic image cache... done"
fi
run_step "[3/4] Starting the web application" \
    compose up -d app nginx

printf "[4/4] Waiting for the web interface... "
attempt=0
until compose exec -T nginx \
    wget -qO- http://127.0.0.1/health >/dev/null 2>&1 &&
    compose exec -T nginx \
        wget -qO- http://127.0.0.1/geo-server/ >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 60 ]; then
        echo "failed"
        compose ps >>"$log_file" 2>&1 || true
        compose logs --no-color --tail=200 >>"$log_file" 2>&1 || true
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
        run_step "Stopping the application" compose stop
        echo "The application is stopped. Its saved data was preserved."
        ;;
    *) echo "The application will keep running in the background." ;;
esac

echo "Use GeoParams again to stop or remove its Docker resources."
