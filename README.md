# Petrographic Image Parametrization

This project analyzes how geologists parameterize and interpret petrographic thin sections.

## Requirements

- PDM (Python package manager)
- Python 3.12
- NVIDIA CUDA driver ≥ 13.0 (required for GPU acceleration only in the
  `ml_moldic_pores` project folder; not needed for `geo_params_web`)
- Visual Studio Code (for running Jupyter notebooks; optional but recommended)

The repository can be partially explored (quick test and data collection
application) without GPU support.

### Dataset

Due to GitHub’s file size limits (100 MB per file), the high-resolution
petrographic thin-section images used in this study cannot be hosted
directly in this repository.

Download the public dataset from Google Drive and place the dataset folders in
the repository root as `datasets/`:
- [https://drive.google.com/drive/folders/1s-NAWbgukQG-1Q3M5MpO808XRqA1QVw4?usp=sharing](https://drive.google.com/drive/folders/1s-NAWbgukQG-1Q3M5MpO808XRqA1QVw4?usp=sharing)

Expected local layout:

```text
datasets/
  article_thin_sections/
  generalization_test_thin_sections/
  pore_type_training/
```

The reduced images required for running the data collection app are generated locally from these files.

The repository contains all code required to process the images once
they are downloaded.

### Running notebooks

All Jupyter Notebooks in this project are prepared to be used from Visual Studio Code.
The included VS Code settings run notebooks from the corresponding subproject
root directory, so relative paths are resolved consistently when the subproject
folder is opened as the workspace.

## Project structure and usage

This repository contains three related but independent subprojects:

- `geo_params_web` — data collection and statistics application
- `ml_moldic_pores` — machine-learning models and analysis notebooks
- `user_params_porosity` — applies collected user parameters to new images
  and exports porosity measurements

Each subproject must be used from its own directory.

When working with the code or notebooks:
- Open Visual Studio Code in the corresponding subproject directory
- Run `./install.sh` inside each subproject directory to create a local PDM environment
- Execute scripts and Jupyter notebooks from within that directory

Python scripts must be executed using `pdm run` to ensure that the correct
project-specific virtual environment is used.

Running commands from the repository root is not supported and may lead
to incorrect paths or missing dependencies.

## Quick test

A minimal quick test is provided for validating the Python environment and
core data structures without requiring images or GPU support.

### Notebook smoke test

From the repository root, run:

```bash
python smoke_test_notebooks.py
```

This validates that the published notebooks can be parsed and that the
expected dataset and generated image-cache paths exist after setup. It does
not execute long-running model-training notebooks.

### Command-line quick test (recommended)

From inside the `geo_params_web` directory, run:

```bash
pdm run python quick_test.py
```

This script loads the processed user-parameterization dataset
(`geo_params_web/static/output/clicks_data.csv`), verifies its structure,
and prints simple statistical summaries. No personal data or image files
are required.

### Notebook-based quick test (optional)

A notebook version of the quick test is also provided in
`geo_params_web/notebooks/quick_test.ipynb` and can be executed from
Visual Studio Code opened inside the `geo_params_web` directory.

### User-parameter porosity test

The `user_params_porosity` subproject provides a direct entry point for
applying the collected user segmentation parameters to petrographic images.
This is useful for reviewers or readers who want to reproduce porosity
measurements without running the web data-collection app or the ML notebooks.

From inside the `user_params_porosity` directory:

```bash
./install.sh
pdm run python quick_test.py
```

This imports the collected user parameters, applies them to one public thin
section, and prints a short porosity summary. The crop metadata is used to
exclude image borders from the measurements.

## Data collection app

The data collection application is located in the `geo_params_web` project folder.

### Docker (recommended)

After placing the public images in `datasets/article_thin_sections/`, run:

```bash
cd geo_params_web
./run.sh
```

The command builds the application image, prepares the local image cache, and
starts the Flask data-collection app and Streamlit statistics app behind an
Nginx reverse proxy:

- Data collection: <http://localhost:8181/geo-server/>
- Statistics: <http://localhost:8181/geo-server/stats/>

The paths match the deployed application layout. Keep the terminal open while
using the applications. Press `Ctrl+C` or close the terminal to stop the
servers. Detailed startup diagnostics are written to
`geo_params_web/log/docker-run.log`.

Set `GEO_PARAMS_PORT` before running the command to use a different host port.
For non-local deployments, also set a strong `FLASK_SECRET_KEY`.

### Local development

1. Open Visual Studio Code from the `geo_params_web` folder.
2. Run `install.sh` to install the application requirements.
3. Run `pdm run python prepare_web_images.py` to generate the local image cache used by the app.
4. Run `run_app_dev.sh` to start the main app.
5. Run `run_stats_sl.sh` to start the statistics app.

## Machine Learning

The machine-learning models are located in the `ml_moldic_pores` project folder.

To work with the machine-learning notebooks:
1. Open Visual Studio Code in the `ml_moldic_pores` project folder
2. Run `./install.sh` to create the local PDM environment
3. Run `pdm run python imports.py` to import the user-parameter table and prepare local ML image derivatives from `datasets/`.
4. Open the notebooks (inside `notebooks` folder) from within Visual Studio Code (the `.vscode` settings are provided; Visual Studio Code may prompt for
installation of the required Jupyter extensions.)

Example usage:

```bash
cd ml_moldic_pores
./install.sh
pdm run python imports.py
code .
```

The main Jupyter notebooks used in the associated paper are:
- `pore-type-supervised-1.2.0.ipynb`
- `pore-type-supervised-1.2.1-stats-json.ipynb`
- `pore-type-supervised-1.2.2-visualizations.ipynb`

## Applying User Parameters to New Images

The `user_params_porosity` subproject is intended for post-collection
porosity measurement. It consumes the anonymized parameter table exported by
`geo_params_web` and applies those $(C_{\min}, K_{\max})$ thresholds to any
set of input images. Optional crop metadata can be supplied as JSON to ensure
that only valid central thin-section areas are measured.

Default analyses are available as short commands:

```bash
cd user_params_porosity
pdm run python run_article_thin_sections.py
pdm run python run_generalization_test.py
```

Outputs are written under `user_params_porosity/data/output/`.

## Related publication

This repository contains the software, notebooks, data, and reproducibility
materials associated with:

Bicudo, M., Tognoli, F., Menasché, D. S., Favoreto, J., and Borghi, L. (2026).
“Through different lenses: Understanding how geologists perceive petrographic
images.” *Computers & Geosciences*, 106222.
[https://doi.org/10.1016/j.cageo.2026.106222](https://doi.org/10.1016/j.cageo.2026.106222)

## Contributing data

This repository contains only anonymized, non-identifiable user data.

If you collect new data using the data collection application and wish
to contribute it to the repository, you must anonymize it before opening
a pull request.

### Required anonymization step

Before submitting a pull request that includes data files:

```bash
cd geo_params_web
pdm run python anonymize_output_for_public_release.py
```

This script removes names, email addresses, free-text feedback, and
developer test sessions. Pull requests containing non-anonymized data
will not be accepted.

## Questions or issues?

Please open an issue in this repository.
