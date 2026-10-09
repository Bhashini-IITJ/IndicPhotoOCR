IMAGE=indicphotoocr
TAG=$(shell date +%Y%m%d)

build:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--cache-from $(IMAGE):latest \
		-t $(IMAGE):latest \
		-t $(IMAGE):$(TAG) \
		.

run:
	docker run \
		--gpus all \
		--rm -it \
		--shm-size=16g \
		-v $(pwd):/workspace \
		--name $(IMAGE) \
		$(IMAGE):latest

build-run: build run