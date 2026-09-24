#!/bin/sh
set -eu

GEOPARAMS_SCRIPT_DIR=$(CDPATH='' cd -P "$(dirname "$0")" && pwd)
export GEOPARAMS_SCRIPT_DIR
# shellcheck source=scripts/lib/common.sh
. "$GEOPARAMS_SCRIPT_DIR/lib/common.sh"

check_run_app() {
    echo "Requirements: run the web application"
    if command -v docker >/dev/null 2>&1 &&
        docker info >/dev/null 2>&1; then
        echo "[OK]      Docker is installed and running"
        if detect_compose; then
            echo "[OK]      Docker Compose is available"
        else
            echo "[MISSING] Docker Compose"
            echo "          $docker_url"
        fi
    else
        echo "[MISSING] Docker"
        echo "          $docker_url"
    fi
    echo "Python and PDM are not required to run the Docker application."
}

python_command=

check_python_312() {
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
                python_command=$candidate
                echo "[OK]      Python 3.12"
                return 0
            fi
        fi
    done
    echo "[MISSING] Python 3.12"
    return 1
}

check_app_development() {
    echo "Requirements: develop the web application"
    check_python_312 ||
        echo "          https://www.python.org/downloads/"
    command_status "PDM" pdm ||
        echo "          https://pdm-project.org/latest/#installation"
    command_status "Docker (recommended for parity)" docker || true
}

check_models() {
    mode=$1
    echo "Requirements: $mode models and research analyses"
    check_python_312 ||
        echo "          https://www.python.org/downloads/"
    command_status "PDM" pdm ||
        echo "          https://pdm-project.org/latest/#installation"
    if [ -d "$repo_dir/datasets/pore_type_training" ]; then
        echo "[FOUND]   Model datasets directory"
    else
        echo "[MISSING] Model datasets; use Dataset management"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[OK]      NVIDIA GPU tools"
    else
        echo "[OPTIONAL] NVIDIA GPU tools for accelerated workflows"
    fi
    if [ "$mode" = develop ]; then
        command_status "Visual Studio Code (optional)" code || true
    fi
}

run_smoke_tests() {
    if check_python_312; then
        (cd "$repo_dir" && "$python_command" smoke_test_notebooks.py)
    else
        echo "Install Python 3.12 manually:"
        echo "https://www.python.org/downloads/"
    fi
}

requirements_menu() {
    while :; do
        echo
        echo "Requirements and tests"
        echo "  1) Check requirements to run the web application"
        echo "  2) Check requirements to develop the web application"
        echo "  3) Check requirements to run models and analyses"
        echo "  4) Check requirements to develop models"
        echo "  5) Run the safe notebook smoke test"
        echo "  b) Back"
        printf "Choice: "
        read -r choice || choice=b
        case "$choice" in
            1) check_run_app ;;
            2) check_app_development ;;
            3) check_models run ;;
            4) check_models develop ;;
            5) run_smoke_tests ;;
            b | B) return 0 ;;
            *) echo "Invalid choice." ;;
        esac
        pause_for_user
    done
}

download_scope() {
    scope=$1
    replace=${2:-false}
    if [ "$scope" = app ]; then
        size="about 329 MB"
    else
        size="about 1.14 GB"
    fi
    echo "This will download $size into: $repo_dir/datasets"
    echo "Every file is accepted only after SHA-256 verification."
    if ! confirm "Continue with this download?"; then
        echo "Download canceled."
        return 0
    fi
    if [ "$replace" = true ]; then
        dataset_tool download --scope "$scope" --replace-invalid
    else
        dataset_tool download --scope "$scope"
    fi
}

datasets_menu() {
    while :; do
        echo
        echo "Dataset management"
        echo "  1) Verify application datasets"
        echo "  2) Download missing application datasets (~329 MB)"
        echo "  3) Verify all research datasets"
        echo "  4) Download all missing research datasets (~1.14 GB)"
        echo "  5) Preserve and replace invalid files"
        echo "  6) Show the manual Google Drive link"
        echo "  b) Back"
        printf "Choice: "
        read -r choice || choice=b
        case "$choice" in
            1) dataset_tool verify --scope app || true ;;
            2) download_scope app ;;
            3) dataset_tool verify --scope all || true ;;
            4) download_scope all ;;
            5)
                echo "Invalid files will be renamed, never silently deleted."
                download_scope all true
                ;;
            6)
                echo "$dataset_url"
                echo "Manual downloads do not require Docker."
                ;;
            b | B) return 0 ;;
            *) echo "Invalid choice." ;;
        esac
        pause_for_user
    done
}

main_menu() {
    while :; do
        echo
        echo "GeoParams repository manager"
        echo "  1) Run or install the web application with Docker"
        echo "  2) Manage and verify public datasets"
        echo "  3) Check requirements and run tests"
        echo "  4) Stop or uninstall the Docker application"
        echo "  q) Quit"
        printf "Choice: "
        read -r choice || choice=q
        case "$choice" in
            1) "$GEOPARAMS_SCRIPT_DIR/run-app.sh" ;;
            2) datasets_menu ;;
            3) requirements_menu ;;
            4) "$GEOPARAMS_SCRIPT_DIR/remove-docker-app.sh" ;;
            q | Q) return 0 ;;
            *) echo "Invalid choice." ;;
        esac
    done
}

main_menu
