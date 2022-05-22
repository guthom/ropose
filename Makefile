MODEL_DOWNLOAD_DIR=./models
DATASETS_DOWNLOAD_DIR=./datasets
HUMAN_SIMPLEBASELINE_GDRIVE_ID=1V7CByQbtsXXDi36xxGqqC9_L-Zvoh9mA

ROPOSE_YOLO_WEIGHTS_PATH=		https://thomas-gulde.de/weights/ropose_yolo.pt
ROPOSE_WEIGHTS_PATH=			https://thomas-gulde.de/weights/ropose_net.pt
ROPOSE_DATASETS_BASE_PATH=		https://thomas-gulde.de/datasets/

.PHONY: prepare-dir
prepare-dir:
	mkdir -p ${MODEL_DOWNLOAD_DIR}
	mkdir -p ${DATASETS_DOWNLOAD_DIR}

.PHONY: download-models
download-models: prepare-dir
	@echo "Downloading ropose_yolo model"
	wget ${ROPOSE_YOLO_WEIGHTS_PATH} -O ${MODEL_DOWNLOAD_DIR}/ropose_yolo.pt
	@echo "Downloading ropose_net model"
	wget ${ROPOSE_WEIGHTS_PATH} -O ${MODEL_DOWNLOAD_DIR}/ropose_net.pt

	@echo "Downloading model for human pose detection (https://arxiv.org/abs/1804.06208)"
	@echo "Please downloads the only the needed weight file from the devs cloud dir -> (https://drive.google.com/drive/folders/1mc1M-hdixlqd7PkqzZiETbPeIAk24WJ-) manually"
	@echo "Place it in ${MODEL_DOWNLOAD_DIR}"
	# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${HUMAN_SIMPLEBASELINE_GDRIVE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${HUMAN_SIMPLEBASELINE_GDRIVE_ID}" -O ${MODEL_DOWNLOAD_DIR}/pose_resnet_152_256x192.pth.tar && rm -rf /tmp/cookies.txt

.PHONY: download-datasets
download-datasets: prepare-dir
	@echo "Downloading train datasets"
	wget ${ROPOSE_DATASETS_BASE_PATH}/ropose_train.7z -O ${DATASETS_DOWNLOAD_DIR}/ropose_train.7z
	@echo "Downloading eval datasets"
	wget ${ROPOSE_DATASETS_BASE_PATH}/colropose_eval.7z -O ${DATASETS_DOWNLOAD_DIR}/colropose_eval.7z
	@echo "Downloading simulated datasets"
	wget ${ROPOSE_DATASETS_BASE_PATH}/ropose_sim.7z -O ${DATASETS_DOWNLOAD_DIR}/ropose_sim.7z

.PHONY: download-all
download-datasets: download-datasets download-models