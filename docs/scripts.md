# Portable Scripts and Launchers

Read this guide before changing scripts, root launchers, or Docker cleanup.
These policies also apply to scripts outside `scripts/`.

## Script Contract

- Prefer POSIX shell and `#!/bin/sh`; document required Bash features.
- Keep `.sh` files LF-terminated and executable.
- Support Git Bash, Linux, and macOS; avoid platform-only flags.
- Check dependencies with `command -v`; never scan whole disks.
  PowerShell uses `Get-Command`, as in the existing Windows launcher.
- Never install host tools; explain missing tools and allow a retry.
- Support both `docker compose` and `docker-compose`.
- Keep Windows launcher behavior aligned with POSIX launchers.
- Remove Docker resources only when both project labels match.

Shared helpers: [common.sh](../scripts/lib/common.sh) and
[common.ps1](../scripts/lib/common.ps1). Root launchers delegate to the managers
in `scripts/`. Cleanup requires both labels below, not just names or image tags:

```text
io.geoparams.project=geo-params-web
io.geoparams.managed-by=geo-params-launcher
```

These labels identify the project, not a deployment. Multiple deployments may
share them; changing cleanup isolation requires a separate design decision.

## Portable Script Tests

- Run ShellCheck, shfmt, Bash, and Dash after shell changes.
- Run mocked flows in Git Bash, WSL, and macOS CI.
- Smoke-test real Docker flows when the local platform permits it.

Use the lint/format arguments and PowerShell parsing checks in
[test-macos.yml](../.github/workflows/test-macos.yml), which covers Ubuntu,
Windows, and macOS. Syntax-check each changed shell file individually with
`bash -n file.sh` and `dash -n file.sh`; multiple filename arguments do not
make these commands check every file.

Run the existing mock harness from the repository root:

```sh
bash tests/shell/test-launchers.sh bash
dash tests/shell/test-launchers.sh dash
bats tests/shell/launchers.bats
```

Run the Bash harness in Git Bash and the Bash/Dash checks in WSL. macOS CI runs
the harness with `sh` and Bash. Report unavailable platforms explicitly;
simulating an OS branch does not replace testing on that platform.

Real Docker smoke checks must use disposable data and isolated resources.
Because cleanup filters by shared labels, a distinct Compose project name
alone does not isolate cleanup from other deployments.
