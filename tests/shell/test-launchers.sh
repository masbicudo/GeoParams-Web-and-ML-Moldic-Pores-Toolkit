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
    mkdir -p "$fixture/mock-bin"
    cp "$repo_dir/run.sh" "$fixture/run.sh"
    cp "$repo_dir/remove-docker-app.sh" "$fixture/remove-docker-app.sh"
    cp "$repo_dir/geo_params_web/run.sh" "$fixture/geo_params_web/run.sh"
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
    chmod +x "$fixture/mock-bin/docker" "$fixture/mock-bin/docker-compose"
}

plugin_fixture="$temp_root/plugin"
make_fixture "$plugin_fixture"
plugin_log="$plugin_fixture/docker.log"
plugin_output="$plugin_fixture/output.log"
: >"$plugin_log"
printf '1\n' | PATH="$plugin_fixture/mock-bin:$PATH" \
    MOCK_LOG="$plugin_log" MOCK_PLUGIN_AVAILABLE=1 \
    "$shell_bin" "$plugin_fixture/run.sh" >"$plugin_output"
assert_contains "$plugin_output" "GeoParams Web is ready"
assert_contains "$plugin_output" "running in the background"
assert_contains "$plugin_log" "docker [compose] [build]"
assert_contains "$plugin_log" "docker [compose] [up] [-d] [app] [nginx]"

standalone_fixture="$temp_root/standalone"
make_fixture "$standalone_fixture"
standalone_log="$standalone_fixture/docker.log"
standalone_output="$standalone_fixture/output.log"
: >"$standalone_log"
printf '2\n' | PATH="$standalone_fixture/mock-bin:$PATH" \
    MOCK_LOG="$standalone_log" MOCK_PLUGIN_AVAILABLE=0 \
    "$shell_bin" "$standalone_fixture/geo_params_web/run.sh" \
    >"$standalone_output"
assert_contains "$standalone_output" "application is stopped"
assert_contains "$standalone_log" "docker-compose [build]"
assert_contains "$standalone_log" "docker-compose [stop]"

cleanup_log="$standalone_fixture/cleanup.log"
cleanup_output="$standalone_fixture/cleanup-output.log"
: >"$cleanup_log"
printf '3\n' | PATH="$standalone_fixture/mock-bin:$PATH" \
    MOCK_LOG="$cleanup_log" \
    "$shell_bin" "$standalone_fixture/remove-docker-app.sh" \
    >"$cleanup_output"
assert_contains "$cleanup_output" "Saved uploads and results were not deleted"
assert_contains "$cleanup_log" "docker [container] [rm] [-f] [container-one]"
assert_contains "$cleanup_log" "docker [container] [rm] [-f] [container-two]"
assert_contains "$cleanup_log" "docker [network] [rm] [network-one]"
assert_contains "$cleanup_log" "docker [image] [rm] [image-one]"
assert_contains "$cleanup_log" "[label=io.geoparams.project=geo-params-web]"
assert_contains "$cleanup_log" "[label=io.geoparams.managed-by=geo-params-launcher]"

echo "Launcher tests passed with $shell_bin."
