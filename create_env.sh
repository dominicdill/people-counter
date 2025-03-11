#!/bin/bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Define variables
ENV_NAME="people-counter"
PYTHON_VERSION="3.13"
YAML_FILE="env-from-history.yml"  # This file contains:
#   name: people-counter
#   channels:
#     - conda-forge
#     - defaults
#   dependencies:
#     - python=3.13
#     - conda-forge::pydantic
#     - conda-forge::pydantic-settings
#     - anaconda::boto3
#     - ipykernel
#     - conda-forge::opencv

# Step 1: Create a minimal conda environment with only Python
conda create -n "$ENV_NAME" python=$PYTHON_VERSION -y
conda init
conda activate "$ENV_NAME"

# Step 2: Install PyTorch via pip (GPU or CPU version)
read -p "Install GPU version of PyTorch? (y/n): " answer
if [ "$answer" = "y" ]; then
    # Change the CUDA version in the URL if needed.
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision torchaudio
fi
pip install ultralytics

# Step 3: Update the environment with the rest of your conda dependencies from the YAML
conda env update --name "$ENV_NAME" --file "$YAML_FILE"

echo "Environment '$ENV_NAME' created successfully."
