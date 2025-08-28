# Use NVIDIA PyTorch as the base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Install additional dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git wget && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for Miniconda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Create the Conda environment and install everything in a single RUN command
RUN conda create -n IndicPhotoOCR-env python=3.9 -y --channel conda-forge --override-channels && \
    git clone https://github.com/Bhashini-IITJ/IndicPhotoOCR.git && \
    cd IndicPhotoOCR && \
    /opt/conda/envs/IndicPhotoOCR-env/bin/python setup.py sdist bdist_wheel && \
    /opt/conda/envs/IndicPhotoOCR-env/bin/pip install ./dist/indicphotoocr-1.3.1-py3-none-any.whl[cu118] --extra-index-url https://download.pytorch.org/whl/cu118

# Set default command to run BharatOCR
# CMD ["/opt/conda/envs/IndicPhotoOCR-env/bin/python", "-m", "IndicPhotoOCR.ocr"]

# To build docker image
# cd IndicPhotoOCR
# sudo docker build -t indicphotoocr:latest .
# sudo docker run --gpus all --rm -it indicphotoocr:latest