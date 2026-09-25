# Development Checks

Read this guide when selecting verification for a change. Inspect the referenced
scripts and configuration for current behavior before running them.

## Python Tests

The root dataset-manager tests use standard-library tools and temporary fixtures.
From the repository root:

```sh
python -m unittest tests.test_dataset_manager -v
```

Use `python3` if that is the available host command, as in Linux CI.
Python subproject commands instead use their own PDM environments.

From `geo_params_web/`, run the separate application unittest suite:

```sh
pdm run python -m unittest discover -s tests -v
```

Its [instructions](../geo_params_web/AGENTS.md) cover focused checks and workflow
regressions. Run both suites for changes affecting shared data contracts or
code used by the application and dataset manager.

## Research and Notebook Tests

From the repository root:

```sh
python smoke_test_notebooks.py
```

This checks notebook JSON, stale paths, and required datasets and generated
files. It does not execute training notebooks. Missing data is a failed
prerequisite, not a passing test. See [setup](../README.md#project-structure-and-usage).

From `geo_params_web/`, `pdm run python quick_test.py` checks the existing
`static/output/clicks_data.csv`; it needs neither images nor a GPU.

From `user_params_porosity/`, `pdm run python quick_test.py` imports parameters,
reads a public image and crop metadata, and writes analysis output. Use a
disposable checkout/data fixture to avoid overwriting real results. See its
[README](../user_params_porosity/README.md) for prerequisites.

`ml_moldic_pores/pyproject.toml` has a `pdm run pytest` alias, but no matching
configured pytest suite or declared pytest dependency was found. Do not treat
this alias or notebook smoke checks as scientific validation. For method
changes, select a relevant reproducible analysis and report inputs and outcome.

## Script Tests

Follow [portable script tests](scripts.md#portable-script-tests).
[test-macos.yml](../.github/workflows/test-macos.yml) defines the existing CI
matrix, shell lint/format checks, mock tests, and PowerShell syntax checks.
The application unittest suite is currently separate from that workflow.

## Documentation Checks

From the repository root, with package dependencies available:

```sh
npm run spellcheck
git diff --check
```

[package.json](../package.json) defines the spelling command;
[cspell.json](../cspell.json) defines dictionaries and exclusions.
For focused spelling checks, pass changed paths to `npx cspell`.
Check relative Markdown links, heading anchors, and new files too; ordinary
`git diff` does not include untracked files.

No repository-wide Python formatter, linter, or type-check command is currently
configured. Commented ML example aliases are not active checks. Do not invent
mandatory tooling as part of an unrelated change.
