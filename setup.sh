#!/bin/bash

# Set environment name and platform variable
ENV_NAME="indicphotoocr"
PLATFORM="cu118"

# Create and activate conda environment
echo "Creating and activating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.9 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Build and install the project
echo "Building and installing IndicPhotoOCR..."
python setup.py sdist bdist_wheel
pip install ./dist/indicphotoocr-1.3.1-py3-none-any.whl[cu118] --extra-index-url https://download.pytorch.org/whl/cu118

echo "Installation complete!"
