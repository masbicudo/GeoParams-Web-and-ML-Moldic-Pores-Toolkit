#!/bin/bash
pdm install
pdm run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# In WSL Ubuntu 22.04 LTS, the above command installs CPU-only torchvision.
# We will upgrade it to the CUDA version here.
if [ "$(uname)" = "Linux" ]; then
    pdm run pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
fi
