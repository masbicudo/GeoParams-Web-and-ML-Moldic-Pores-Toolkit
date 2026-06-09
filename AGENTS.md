# AGENTS.md

Guidance for future coding agents working in this repository.

## Repository Shape

This repository is the public, reproducible artifact for the manuscript. Keep
it narrower and cleaner than the historical research sandbox.

Subprojects:

- `geo_params_web` - data-collection and statistics app.
- `ml_moldic_pores` - machine-learning notebooks and models.
- `user_params_porosity` - post-collection porosity measurements using user
  parameters.

Each subproject has its own PDM environment and should be run from its own
directory.

## Data Layout

Large image datasets live in the repository-level `datasets/` directory and
are ignored by Git:

```text
datasets/
  article_thin_sections/
  generalization_test_thin_sections/
  pore_type_training/
```

Small metadata that defines a method, such as crop rectangles, can be
versioned. Large images, generated outputs, caches, and derived analysis tables
should normally stay ignored unless they are final manuscript artifacts.

## Derived Data

Treat subproject data/output directories as local caches or generated outputs:

- `geo_params_web/static/imgs_sections/` is a cache for the Flask app.
- `ml_moldic_pores/out/` is a cache for ML-ready derived images/models.
- `user_params_porosity/data/output/` is generated porosity output.

Do not make a generated cache the conceptual source of truth when a dataset or
metadata file can be referenced directly.

## Notebooks and Paths

VS Code is configured to run notebooks from the subproject workspace root with:

```json
"jupyter.notebookFileRoot": "${workspaceFolder}"
```

New notebooks and scripts should use paths relative to the subproject root, or
resolve paths from `Path(__file__)` for importable modules. Avoid relying on
the notebook file's directory as the current working directory.

## Reader Experience

Prefer short, safe commands in README entry points. A reviewer or reader should
be able to run a smoke test without copying long path-heavy commands.

Good first-contact commands look like:

```bash
pdm run python quick_test.py
pdm run python run_generalization_test.py
```

Keep long configurable commands in "Advanced Usage" sections.

CLI failures should explain the missing path and the expected dataset layout
instead of exposing a long traceback as the first user-facing experience.

## Scope Control

Do not import experimental methodology from external collaboration folders
unless the manuscript explicitly needs it. Improvements developed in other
collaborations can be mentioned as future work, but the public artifact should
preserve the method used in the manuscript unless the manuscript changes too.
