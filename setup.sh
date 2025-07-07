#!/bin/bash

# Set environment name and platform variable
ENV_NAME="indicphotoocr"
PLATFORM="cu118"

# Create and activate conda environment
echo "Creating and activating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.9 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# # Clone the repository and navigate into it
# echo "Cloning the IndicPhotoOCR repository..."
# git clone https://github.com/Bhashini-IITJ/IndicPhotoOCR.git
# cd IndicPhotoOCR || exit

# Install pip-tools and generate requirements files
#echo "Installing pip-tools and generating requirements files..."
#pip install pip-tools
#make clean-reqs reqs

# Generate requirements files for specified PyTorch platform
#echo "Generating PyTorch requirements for platform: $PLATFORM"
#make torch-"${PLATFORM}"

# Install project dependencies
#echo "Installing project dependencies..."
#pip install -r requirements/core."${PLATFORM}".txt -e .[train,test]
#pip install opencv-python==4.10.0.84
#pip install shapely==2.0.6
#pip install openai-clip==1.0.1
#pip install lmdb==1.5.1

# Build and install the project
echo "Building and installing IndicPhotoOCR..."
python setup.py sdist bdist_wheel
pip install ./dist/IndicPhotoOCR-1.3.0-py3-none-any.whl[cu118] --extra-index-url https://download.pytorch.org/whl/cu118

echo "Installation complete!"
