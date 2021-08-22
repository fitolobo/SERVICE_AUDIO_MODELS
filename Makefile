#FUNCTION
define cecho
    @echo "\033[92m$(1)\033[0m"
endef

# VARIABLES
REGISTRY_URL=fitolobo
VERSION=latest
DOCKER_IMAGE=audio-app-binary-emotions
DOCKER_FILE=Dockerfile

##############################################################################
############################# DOCKER VARS ####################################
##############################################################################
# COMMANDS
DOCKER_COMMAND=docker

#HOST VARS
LOCALHOST_IP=127.0.0.1
HOST_TENSORBOARD_PORT=26006
HOST_NOTEBOOK_PORT=28888
HOST_SERVER_PORT=8000

#HOST CPU VARS
HOST_SOURCE_PATH=$(shell pwd)

#IMAGE VARS
IMAGE_TENSORBOARD_PORT=6006
IMAGE_NOTEBOOK_PORT=8888
IMAGE_SOURCE_PATH=/home/src
IMAGE_SERVER_PORT=8000
# VOLUMES

DOCKER_VOLUMES = --volume=$(HOST_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
		     				 --workdir=$(IMAGE_SOURCE_PATH) \
		     				 --shm-size 8G

DOCKER_PORTS= -p $(LOCALHOST_IP):$(HOST_SERVER_PORT):$(IMAGE_SERVER_PORT)
DOCKER_TENSORBOARD_PORTS = -p $(LOCALHOST_IP):$(HOST_TENSORBOARD_PORT):$(IMAGE_TENSORBOARD_PORT)
DOCKER_JUPYTER_PORTS = -p $(LOCALHOST_IP):$(HOST_NOTEBOOK_PORT):$(IMAGE_NOTEBOOK_PORT)

DOCKER_RUN_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host $(DOCKER_PORTS) $(DOCKER_VOLUMES) $(REGISTRY_URL)/$(DOCKER_IMAGE):$(VERSION)
DOCKER_RUN_TENSORBOARD_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host  $(DOCKER_TENSORBOARD_PORTS) $(DOCKER_VOLUMES) $(REGISTRY_URL)/$(DOCKER_IMAGE):$(VERSION)
DOCKER_RUN_JUPYTER_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host  $(DOCKER_JUPYTER_PORTS) $(DOCKER_VOLUMES) $(REGISTRY_URL)/$(DOCKER_IMAGE):$(VERSION)


# COMMANDS
JUPYTER_COMMAND=jupyter
TENSORBOARD_COMMAND=tensorboard
MKDIR_COMMAND=mkdir
WGET_COMMAND=wget

# URLs
TENSORBOARD_PATH=$(IMAGE_METADATA_PATH)

run rtm: docker-print
	@$(DOCKER_RUN_COMMAND)

jupyter jp:
	$(call cecho, "[Jupyter] Running Jupyter lab")
	@$(EXPORT_COMMAND)
	@$(JUPYTER_COMMAND) lab --ip=0.0.0.0 --allow-root

run-jupyter rj: docker-print
	@$(DOCKER_RUN_JUPYTER_COMMAND)  bash -c "make jupyter"; \
	status=$$?


tensorboard tb:
	$(call cecho, "[Tensorboard] Running Tensorboard")
	@$(TENSORBOARD_COMMAND) --logdir=$(TENSORBOARD_PATH) --host 0.0.0.0

run-tensorboard rt: docker-print
	@$(DOCKER_RUN_TENSORBOARD_COMMAND)  bash -c "make tensorboard TENSORBOARD_PATH=$(TENSORBOARD_PATH)"; \
	status=$$?


build b:
	@echo "[build] Building cpu docker image..."
	@$(DOCKER_COMMAND) build -t $(REGISTRY_URL)/$(DOCKER_IMAGE):$(VERSION) -f $(DOCKER_FILE) .
	@echo "[build] Delete old versions..."
	@$(DOCKER_COMMAND) images|sed "1 d"|grep "<none> *<none>"|awk '{print $$3}'|sort|uniq|xargs $(DOCKER_COMMAND) rmi -f

#PRIVATE
docker-print psd:
	$(call cecho, "[Docker] Running docker image...")
