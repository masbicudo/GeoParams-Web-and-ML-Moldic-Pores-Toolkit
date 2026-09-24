# AGENTS.md

## About This File

- Human developers and coding agents MUST read and follow this file.
- This section MUST remain the first section in this file.
- Interpret MUST, SHALL, SHOULD, and MAY per RFC 2119 and RFC 8174.
- Rules SHOULD fit within 80 characters; rare exceptions MUST aid clarity.
- Rules SHOULD remain concise, specific, and nonredundant.
- Order sections and their rules by relevance to human developers.

## Branches

- `main` is the research branch associated with the publication.
- Keep `main` suitable for readers reproducing the published work.
- `publication-cageo-2026` is the exact published baseline.
- `app-v2` is the integration branch for active app development.
- Start app features on short-lived branches based on `app-v2`.
- Merge validated app features into `app-v2` with small commits.
- Never merge all of `app-v2` into `main`.
- Port fixes to `main` only when they suit the research artifact.
- Keep the branch READMEs distinct and link them to each other.
- Explain in `main` that app development continues on `app-v2`.

## Privacy

- Only personal data is sensitive by default.
- Do not expose personal data in public lists, summaries, or examples.
- Do not use personal names in commit messages.
- Dataset names and filenames may appear in lists and summaries.

## Data

- Keep large datasets in the ignored repository-level `datasets/`.
- Version small metadata when it defines a reproducible method.
- Keep uploads, caches, generated results, and credentials out of Git.
- Store persistent app data in a bind-mounted, ignored host directory.
- Never rely on a Docker container layer for persistent user data.
- Treat generated outputs as caches, not as source datasets.

## Repository

- Keep this public repository reproducible and easy to inspect.
- `geo_params_web` contains the collection and statistics app.
- `ml_moldic_pores` contains ML notebooks and models.
- `user_params_porosity` measures porosity from user parameters.
- Run each subproject from its own PDM environment and directory.

## Scripts

- Prefer POSIX shell and `#!/bin/sh`; document required Bash features.
- Keep `.sh` files LF-terminated and executable.
- Support Git Bash, Linux, and macOS; avoid platform-only flags.
- Check dependencies with `command -v`; never scan whole disks.
- Never install host tools; explain missing tools and allow a retry.
- Support both `docker compose` and `docker-compose`.
- Keep Windows launcher behavior aligned with POSIX launchers.
- Remove Docker resources only when both project labels match.
- Run ShellCheck, shfmt, Bash, and Dash after shell changes.
- Run mocked flows in Git Bash, WSL, and macOS CI.
- Smoke-test real Docker flows when the local platform permits it.

## Paths and Commands

- Resolve script paths from the subproject root or `Path(__file__)`.
- Do not depend on a notebook file's directory as the working directory.
- Keep README entry commands short and safe to copy.
- Put long configurable commands in an advanced section.
- Report missing paths and expected layouts without noisy tracebacks.

## Workflows

- A tool flow may include human steps and automated jobs.
- Add regression tests for concurrent queues and workflow state changes.
- Coordinate concurrency tests with events or barriers, not timing sleeps.
- Human input must not hold an automated processing slot.
- Show automated progress as part of its parent tool flow.
- Serialize jobs that compete for the same processing resource.
- Mark flows waiting for user input without blocking queued jobs.
- Make completed parameter collections read-only.
- Open completed collections on a summary page.
- Put flow lists on tool landing pages, not on result pages.
- Show active flows from all tools in the global helper.
- Persist job state and results in the mounted data directory.
