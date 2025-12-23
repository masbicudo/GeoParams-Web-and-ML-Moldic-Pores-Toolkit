# Petrographic Image Parametrization

This project analyzes how geologists parameterize and interpret petrographic thin sections.

## Requirements

- PDM (Python package manager)
- Python 3.12
- NVIDIA CUDA driver ≥ 13.0 (required for GPU acceleration only in the
  `ml_moldic_pores` project folder; not needed for `geo_params_web`)
- Visual Studio Code

The repository can be partially explored (quick test and data collection
application) without GPU support.

### Dataset

Due to GitHub’s file size limits (100 MB per file), the high-resolution
petrographic thin-section images used in this study cannot be hosted
directly in this repository.

Copy the thin-section files from `GoogleDrive/data` to `ml_moldic_pores/data`:
- [https://drive.google.com/drive/folders/1s-NAWbgukQG-1Q3M5MpO808XRqA1QVw4?usp=sharing](https://drive.google.com/drive/folders/1s-NAWbgukQG-1Q3M5MpO808XRqA1QVw4?usp=sharing)

The reduced images required for running the data collection app are generated locally from these files.

The repository contains all code required to process the images once
they are downloaded.

### Running notebooks

All Jupyter Notebooks in this project are prepared to be used from Visual Studio Code.

## Project structure and usage

This repository contains two related but independent subprojects:

- `geo_params_web` — data collection and statistics application
- `ml_moldic_pores` — machine-learning models and analysis notebooks

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

## Data collection app

The data collection application is located in the `geo_params_web` project folder.

1. Open Visual Studio Code from the `geo_params_web` folder.
2. Open the notebook at `notebooks/make-images.ipynb` from within VS Code and run all cells to generate the reduced images used by the app.
3. Run `install.sh` to install the application requirements.
4. Run `run_app_dev.sh` to start the main app.
5. Run `run_stats_sl.sh` to start the statistics app.

## Machine Learning

The machine-learning models are located in the `ml_moldic_pores` project folder.

To work with the machine-learning notebooks:
1. Open Visual Studio Code in the `ml_moldic_pores` project folder
2. Run `./install.sh` to create the local PDM environment
3. Open the notebooks (inside `notebooks` folder) from within Visual Studio Code (the `.vscode` settings are provided; Visual Studio Code may prompt for
installation of the required Jupyter extensions.)

Example usage:

```bash
cd ml_moldic_pores
./install.sh
code .
```

The main Jupyter notebooks used in the associated paper are:
- `pore-type-supervised-1.2.0.ipynb`
- `pore-type-supervised-1.2.1-stats-json.ipynb`
- `pore-type-supervised-1.2.2-visualizations.ipynb`

## Related research

This repository accompanies an unpublished research manuscript on the
perception and parameterization of petrographic thin sections by geologists.

The code and notebooks provided here were used to generate figures,
analyses, and results described in that manuscript.

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
