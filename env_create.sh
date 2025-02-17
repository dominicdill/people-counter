#!/bin/bash

# Extract the environment name from env.yml (the second word on the line containing "name:")
ENV_NAME=$(awk '/name:/ {print $2}' env.yml)

# Check if conda is installed and available in the PATH
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in the PATH."
    exit 1
fi

# Check if the environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    # Prompt the user for a decision
    read -p "Environment '$ENV_NAME' already exists. Do you want to remove and recreate it? [y/n]: " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "Removing existing environment: $ENV_NAME"
        conda env remove -n "$ENV_NAME"
        echo "Creating new environment from env.yml..."
        conda env create -f env.yml
    else
        echo "Using the existing environment: $ENV_NAME"
    fi
else
    echo "Creating new environment from env.yml..."
    conda env create -f env.yml
fi

# Activate the environment
echo "Activating the environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Conda environment '$ENV_NAME' is now active."
