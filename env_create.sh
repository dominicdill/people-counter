#!/bin/bash

ENV_NAME=$(awk '/name:/ {print $2}' env.yml)

if ! command -v conda &> /dev/null
then
    echo "Error: Conda is not installed or not in the PATH."
    exit 1
fi

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Removing existing environment: $ENV_NAME"
    conda env remove -n "$ENV_NAME"
fi

echo "Creating new environment from env.yml..."
conda env create -f env.yml

echo "Activating the environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Conda environment '$ENV_NAME' is now active."
