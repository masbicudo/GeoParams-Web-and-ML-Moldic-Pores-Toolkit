# AGENTS.md

Guidance for coding agents working in this repository.

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

## Repository

- Keep this public repository reproducible and easy to inspect.
- `geo_params_web` contains the collection and statistics app.
- `ml_moldic_pores` contains ML notebooks and models.
- `user_params_porosity` measures porosity from user parameters.
- Run each subproject from its own PDM environment and directory.

## Data

- Keep large datasets in the ignored repository-level `datasets/`.
- Version small metadata when it defines a reproducible method.
- Keep uploads, caches, generated results, and credentials out of Git.
- Store persistent app data in a bind-mounted, ignored host directory.
- Never rely on a Docker container layer for persistent user data.
- Treat generated outputs as caches, not as source datasets.

## Privacy

- Only personal data is sensitive by default.
- Do not expose personal data in public lists, summaries, or examples.
- Do not use personal names in commit messages.
- Dataset names and filenames may appear in lists and summaries.

## Workflows

- A tool flow may include human steps and automated jobs.
- Human input must not hold an automated processing slot.
- Show automated progress as part of its parent tool flow.
- Serialize jobs that compete for the same processing resource.
- Mark flows waiting for user input without blocking queued jobs.
- Make completed parameter collections read-only.
- Open completed collections on a summary page.
- Put flow lists on tool landing pages, not on result pages.
- Show active flows from all tools in the global helper.
- Persist job state and results in the mounted data directory.

## Paths and Commands

- Resolve script paths from the subproject root or `Path(__file__)`.
- Do not depend on a notebook file's directory as the working directory.
- Keep README entry commands short and safe to copy.
- Put long configurable commands in an advanced section.
- Report missing paths and expected layouts without noisy tracebacks.
