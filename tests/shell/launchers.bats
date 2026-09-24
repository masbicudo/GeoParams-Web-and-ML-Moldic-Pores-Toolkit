#!/usr/bin/env bats

@test "portable launchers work with Bash" {
    run bash "$BATS_TEST_DIRNAME/test-launchers.sh" bash
    [ "$status" -eq 0 ]
}

@test "portable launchers work with Dash" {
    command -v dash >/dev/null 2>&1 || skip "Dash is not installed"
    run dash "$BATS_TEST_DIRNAME/test-launchers.sh" dash
    [ "$status" -eq 0 ]
}
