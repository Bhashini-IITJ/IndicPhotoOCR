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

# Install the package from the build context (this checkout), so the image
# runs exactly the code being built (default detector: fine-tuned YOLO11n).
WORKDIR /workspace
COPY . /workspace/IndicPhotoOCR
RUN pip install --no-cache-dir -e IndicPhotoOCR \
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
# Inside the container:
# cd /workspace/IndicPhotoOCR && python end-to-end-Inference.py -p <images> -o predictions.json