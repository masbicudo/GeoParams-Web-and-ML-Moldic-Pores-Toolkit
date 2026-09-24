#!/bin/sh
set -eu

shell_bin=${1:-sh}
if ! command -v "$shell_bin" >/dev/null 2>&1; then
    echo "Shell not found: $shell_bin" >&2
    exit 1
fi

script_dir=$(CDPATH='' cd -P "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH='' cd -P "$script_dir/../.." && pwd)
temp_root=$(mktemp -d "${TMPDIR:-/tmp}/geoparams-test.XXXXXX")

cleanup() {
    if [ -n "${temp_root:-}" ] && [ -d "$temp_root" ]; then
        rm -rf "$temp_root"
    fi
}
trap cleanup EXIT HUP INT TERM

fail() {
    echo "FAIL: $1" >&2
    exit 1
}

assert_contains() {
    file=$1
    expected=$2
    grep -F "$expected" "$file" >/dev/null 2>&1 ||
        fail "Expected '$expected' in $file"
}

make_fixture() {
    fixture=$1
    mkdir -p "$fixture/geo_params_web/static/imgs_sections"
    mkdir -p "$fixture/scripts/lib" "$fixture/mock-bin"
    cp "$repo_dir/GeoParams.sh" "$fixture/GeoParams.sh"
    cp "$repo_dir/GeoParams.command" "$fixture/GeoParams.command"
    cp "$repo_dir/scripts/manager.sh" "$fixture/scripts/manager.sh"
    cp "$repo_dir/scripts/run-app.sh" "$fixture/scripts/run-app.sh"
    cp "$repo_dir/scripts/remove-docker-app.sh" \
        "$fixture/scripts/remove-docker-app.sh"
    cp "$repo_dir/scripts/lib/common.sh" "$fixture/scripts/lib/common.sh"
    : >"$fixture/geo_params_web/static/imgs_sections/sample.jpg"
    printf '{}\n' >"$fixture/geo_params_web/static/imgs_sections/metadata.json"

    cat >"$fixture/mock-bin/docker" <<'EOF'
#!/bin/sh
printf 'docker' >>"$MOCK_LOG"
for argument in "$@"; do
    printf ' [%s]' "$argument" >>"$MOCK_LOG"
done
printf '\n' >>"$MOCK_LOG"

if [ "${1:-}" = compose ] && [ "${2:-}" = version ]; then
    [ "${MOCK_PLUGIN_AVAILABLE:-1}" = 1 ]
    exit
fi
if [ "${1:-}" = info ]; then
    exit 0
fi
if [ "${1:-}" = container ] && [ "${2:-}" = ls ]; then
    printf 'container-one\ncontainer-two\n'
    exit 0
fi
if [ "${1:-}" = network ] && [ "${2:-}" = ls ]; then
    printf 'network-one\n'
    exit 0
fi
if [ "${1:-}" = image ] && [ "${2:-}" = ls ]; then
    printf 'image-one\nimage-one\n'
    exit 0
fi
exit 0
EOF

    cat >"$fixture/mock-bin/docker-compose" <<'EOF'
#!/bin/sh
printf 'docker-compose' >>"$MOCK_LOG"
for argument in "$@"; do
    printf ' [%s]' "$argument" >>"$MOCK_LOG"
done
printf '\n' >>"$MOCK_LOG"
exit 0
EOF
    chmod +x "$fixture/mock-bin/docker" \
        "$fixture/mock-bin/docker-compose"
}

plugin_fixture="$temp_root/plugin"
make_fixture "$plugin_fixture"
plugin_log="$plugin_fixture/docker.log"
plugin_output="$plugin_fixture/output.log"
: >"$plugin_log"
printf '1\n1\nq\n' | PATH="$plugin_fixture/mock-bin:$PATH" \
    MOCK_LOG="$plugin_log" MOCK_PLUGIN_AVAILABLE=1 \
    "$shell_bin" "$plugin_fixture/GeoParams.sh" >"$plugin_output"
assert_contains "$plugin_output" "GeoParams Web is ready"
assert_contains "$plugin_output" "running in the background"
assert_contains "$plugin_log" "docker [compose] [build] [app]"
assert_contains "$plugin_log" "docker [compose] [up] [-d] [app] [nginx]"

standalone_fixture="$temp_root/standalone"
make_fixture "$standalone_fixture"
standalone_log="$standalone_fixture/docker.log"
standalone_output="$standalone_fixture/output.log"
: >"$standalone_log"
printf '2\n' | PATH="$standalone_fixture/mock-bin:$PATH" \
    MOCK_LOG="$standalone_log" MOCK_PLUGIN_AVAILABLE=0 \
    "$shell_bin" "$standalone_fixture/scripts/run-app.sh" \
    >"$standalone_output"
assert_contains "$standalone_output" "application is stopped"
assert_contains "$standalone_log" "docker-compose [build] [app]"
assert_contains "$standalone_log" "docker-compose [stop]"

cleanup_log="$standalone_fixture/cleanup.log"
cleanup_output="$standalone_fixture/cleanup-output.log"
: >"$cleanup_log"
printf '3\n' | PATH="$standalone_fixture/mock-bin:$PATH" \
    MOCK_LOG="$cleanup_log" MOCK_PLUGIN_AVAILABLE=1 \
    "$shell_bin" "$standalone_fixture/scripts/remove-docker-app.sh" \
    >"$cleanup_output"
assert_contains "$cleanup_output" "downloaded datasets were not deleted"
assert_contains "$cleanup_log" "docker [container] [rm] [-f] [container-one]"
assert_contains "$cleanup_log" "docker [container] [rm] [-f] [container-two]"
assert_contains "$cleanup_log" "docker [network] [rm] [network-one]"
assert_contains "$cleanup_log" "docker [image] [rm] [image-one]"
assert_contains "$cleanup_log" "[label=io.geoparams.project=geo-params-web]"
assert_contains "$cleanup_log" \
    "[label=io.geoparams.managed-by=geo-params-launcher]"

menu_output="$plugin_fixture/menu-output.log"
printf '2\n6\n\nb\nq\n' | PATH="$plugin_fixture/mock-bin:$PATH" \
    MOCK_LOG="$plugin_log" MOCK_PLUGIN_AVAILABLE=1 \
    "$shell_bin" "$plugin_fixture/GeoParams.command" >"$menu_output"
assert_contains "$menu_output" "drive.google.com/drive/folders"
assert_contains "$menu_output" "Manual downloads do not require Docker"

echo "Launcher tests passed with $shell_bin."
