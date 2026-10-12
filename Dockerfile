# Use NVIDIA PyTorch as the base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Standard Python/pip hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Clone and install the package
WORKDIR /workspace
RUN git clone https://github.com/dikshant-sharma05/IndicPhotoOCR.git && \
    pip install --no-cache-dir -e IndicPhotoOCR \
        --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python && \
    pip install --no-cache-dir opencv-python-headless==4.7.0.72

CMD ["bash"]


# Set default command to run BharatOCR
# CMD ["/opt/conda/envs/IndicPhotoOCR-env/bin/python", "-m", "IndicPhotoOCR.ocr"]

# To build docker image
# cd IndicPhotoOCR
# sudo docker build -t indicphotoocr:latest .
# sudo docker run --gpus all --rm -it indicphotoocr:latest