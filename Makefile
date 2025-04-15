install:
	uv sync

clean:
	rm -rf dist/ yolo_training.egg-info/

.PHONY: build
build:
	uv build --sdist

publish: clean build
	viam training-script upload --path dist/yolo_training-$(version).tar.gz --org-id 16518049-9dd3-479f-9644-c5d112aa42d8 --framework onnx --script-name=yolo-onnx-training --type object_detection --version $(version)

all: clean build
