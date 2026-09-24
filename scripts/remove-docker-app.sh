#!/bin/sh
set -eu

GEOPARAMS_SCRIPT_DIR=$(CDPATH='' cd -P "$(dirname "$0")" && pwd)
export GEOPARAMS_SCRIPT_DIR
# shellcheck source=scripts/lib/common.sh
. "$GEOPARAMS_SCRIPT_DIR/lib/common.sh"

project_label="io.geoparams.project=geo-params-web"
manager_label="io.geoparams.managed-by=geo-params-launcher"

ensure_docker

container_ids() {
    docker container ls -aq --filter "label=$project_label" \
        --filter "label=$manager_label"
}

stop_project_containers() {
    ids=$(docker container ls -q --filter "label=$project_label" \
        --filter "label=$manager_label")
    if [ -z "$ids" ]; then
        echo "No running project containers were found."
        return
    fi
    while IFS= read -r id; do
        [ -n "$id" ] && docker container stop "$id"
    done <<EOF
$ids
EOF
}

remove_project_containers() {
    ids=$(container_ids)
    if [ -z "$ids" ]; then
        echo "No project-labeled containers were found."
        return
    fi
    while IFS= read -r id; do
        [ -n "$id" ] && docker container rm -f "$id"
    done <<EOF
$ids
EOF
}

remove_project_networks() {
    ids=$(docker network ls -q --filter "label=$project_label" \
        --filter "label=$manager_label")
    if [ -z "$ids" ]; then
        echo "No project-labeled networks were found."
        return
    fi
    while IFS= read -r id; do
        [ -n "$id" ] && docker network rm "$id"
    done <<EOF
$ids
EOF
}

echo "GeoParams Web - Docker cleanup"
echo "  1) Stop the application and keep its containers"
echo "  2) Remove its containers and private network"
echo "  3) Also remove images built and labeled by this project"
echo "  q) Cancel"
printf "Choice: "
read -r choice || choice=q

case "$choice" in
    1) stop_project_containers ;;
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

echo "Saved uploads, results, and downloaded datasets were not deleted."
