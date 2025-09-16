# Makefileの存在するディレクトリ
MAKEFILE_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# 一つ上のディレクトリ
PARENT_DIR := $(shell dirname ${MAKEFILE_DIR})

build:
	bash ${PARENT_DIR}/docker-environments/gaussian_splatting_train/build_script.sh

up:
	bash ${PARENT_DIR}/docker-environments/gaussian_splatting_train/up_script.sh $(DISPLAY) my-unet-env

in:
	docker start my-unet-env
	docker exec -it my-unet-env bash

allow-gui-host:
	xhost + local:

setup:
	conda env create --file environment.yml

active:
	conda activate my_unet

up-jupyter: 
	jupyter lab --ip='0.0.0.0' --port=40203 --no-browser --allow-root
