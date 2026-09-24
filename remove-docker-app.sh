#!/bin/sh
set -eu

repo_dir=$(CDPATH='' cd -P "$(dirname "$0")" && pwd)
app_dir="$repo_dir/geo_params_web"
project_label="io.geoparams.project=geo-params-web"
manager_label="io.geoparams.managed-by=geo-params-launcher"

if ! command -v docker >/dev/null 2>&1 ||
    ! docker info >/dev/null 2>&1; then
    echo "Docker is unavailable. Start it before managing the application."
    exit 1
fi

cd "$app_dir"

container_ids() {
    docker container ls -aq --filter "label=$project_label" \
        --filter "label=$manager_label"
}

stop_project_containers() {
    ids=$(docker container ls -q --filter "label=$project_label" \
        --filter "label=$manager_label")
    if [ -n "$ids" ]; then
        while IFS= read -r id; do
            [ -n "$id" ] && docker container stop "$id"
        done <<EOF
$ids
EOF
    else
        echo "No running project containers were found."
    fi
}

remove_project_containers() {
    ids=$(container_ids)
    if [ -n "$ids" ]; then
        while IFS= read -r id; do
            [ -n "$id" ] && docker container rm -f "$id"
        done <<EOF
$ids
EOF
    else
        echo "No project-labeled containers were found."
    fi
}

remove_project_networks() {
    ids=$(docker network ls -q --filter "label=$project_label" \
        --filter "label=$manager_label")
    if [ -n "$ids" ]; then
        while IFS= read -r id; do
            [ -n "$id" ] && docker network rm "$id"
        done <<EOF
$ids
EOF
    else
        echo "No project-labeled networks were found."
    fi
}

echo "GeoParams Web - Docker cleanup"
echo "  1) Stop the application and keep its containers"
echo "  2) Remove its containers and private network"
echo "  3) Also remove images built and labeled by this project"
echo "  q) Cancel"
printf "Choice: "
read -r choice || choice=q

case "$choice" in
    1)
        stop_project_containers
        ;;
    2)
        remove_project_containers
        remove_project_networks
        ;;
    3)
        remove_project_containers
        remove_project_networks
        image_ids=$(docker image ls --filter "label=$project_label" \
            --filter "label=$manager_label" --format '{{.ID}}' | sort -u)
        if [ -n "$image_ids" ]; then
            while IFS= read -r image_id; do
                [ -n "$image_id" ] && docker image rm "$image_id"
            done <<EOF
$image_ids
EOF
        else
            echo "No project-labeled images were found."
        fi
        ;;
    q | Q)
        echo "Nothing was changed."
        exit 0
        ;;
    *)
        echo "Invalid choice. Nothing was changed."
        exit 1
        ;;
esac

echo "Saved uploads and results were not deleted."
