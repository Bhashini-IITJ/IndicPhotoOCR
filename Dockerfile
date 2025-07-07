# Use NVIDIA PyTorch as the base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Install additional dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# Set environment variables for Miniconda and Conda environment
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Create a new Conda environment named "bocr" with Python 3.9
RUN conda create -n bocr python=3.9 -y

# Initialize conda 
RUN conda init

# Reload the env configs
RUN source ~/.bashrc

# Make RUN commands use the bocr environment
SHELL ["conda", "run", "-n", "bocr", "/bin/bash", "-c"]

# # Set default shell to bash
# SHELL ["/bin/bash", "-c"]

# # Clone BharatOCR repository
# RUN git clone https://github.com/Bhashini-IITJ/IndicPhotoOCR.git && \
#     cd IndicPhotoOCR && \
#     git switch photoOCR && \
#     python setup.py sdist bdist_wheel && \
#     pip install ./dist/IndicPhotoOCR-1.2.0-py3-none-any.whl[cu118] --extra-index-url https://download.pytorch.org/whl/cu118

# # # Set default command to run BharatOCR
# CMD ["conda", "run", "-n", "bocr", "python", "-m", "IndicPhotoOCR.ocr"]


# cd IndicPhotoOCR
# sudo docker build -t indicphotoocr:latest .
# sudo docker run --gpus all --rm -it indicphotoocr:latest