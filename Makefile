IMAGE ?= indicphotoocr
TAG ?= $(shell date +%Y%m%d)

# The checkout is mounted at /workspace/IndicPhotoOCR, so the container runs the
# current code and downloaded model weights persist between runs.
DOCKER_RUN = docker run --gpus all --rm --shm-size=16g \
	-v $(CURDIR):/workspace/IndicPhotoOCR

.PHONY: build run test build-run

build:
	docker build -t $(IMAGE):latest -t $(IMAGE):$(TAG) .

run:
	$(DOCKER_RUN) -it --name $(IMAGE) $(IMAGE):latest

test:
	$(DOCKER_RUN) $(IMAGE):latest python -m pytest -p no:cacheprovider tests/

build-run: build run
