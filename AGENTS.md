# AGENTS.md

## About This File

- Human developers and coding agents MUST read and follow this file.
- This section MUST remain the first section in this file.
- Interpret MUST, SHALL, SHOULD, and MAY per RFC 2119 and RFC 8174.
- Rules SHOULD fit within 80 characters; rare exceptions MUST aid clarity.
- Rules SHOULD remain concise, specific, and nonredundant.
- Order sections and their rules by relevance to human developers.
- Read nested AGENTS.md files before working in their scope.
- Nested rules inherit this contract; do not repeat inherited rules.

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
- Never rely on a Docker container layer for persistent user data.
- Treat generated outputs as caches, not as source datasets.

## Repository

- Keep this public repository reproducible and easy to inspect.
- `geo_params_web` contains the collection and statistics app.
- `ml_moldic_pores` contains ML notebooks and models.
- `user_params_porosity` measures porosity from user parameters.
- Run each subproject from its own PDM environment and directory.

## Development Workflow

- Inspect relevant code and configuration before editing.
- Use existing repository scripts and tools; derive commands from files.
- Fix regressions introduced by your changes.

### General Tests

- Run narrow automated checks while iterating; broaden them with change scope.
- Run the full suite before merging.
- Keep tests deterministic, isolated, and independent of execution order.
- Use temporary directories; never alter real uploads or results.
- Report executed checks, results, and unavailable checks explicitly.
- Do not substitute mental inference for available automated verification.

## Paths and Commands

- Resolve script paths from the subproject root or `Path(__file__)`.
- Do not depend on a notebook file's directory as the working directory.
- Keep README entry commands short and safe to copy.
- Put long configurable commands in an advanced section.
- Report missing paths and expected layouts without noisy tracebacks.

## Task Routing

- For app work, read [app rules](geo_params_web/AGENTS.md).
- For scientific method work, read [method docs](user_params_porosity/README.md).
- For scripts or Docker cleanup, read [script policies](docs/scripts.md).
- For verification, read [development checks](docs/development-checks.md).
- For TODO changes, follow [the convention](docs/about-todos.md).
