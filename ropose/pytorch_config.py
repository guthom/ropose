import os, sys
from guthoms_helpers.common_helpers.ColorHelper import ColorHelper
import math
import numpy as np
import torch

from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper

#path stuff
homedir = os.environ['HOME']
projectDir = os.path.dirname(os.path.realpath(__file__))

# append third party stuff
thirdpartyBaseLinePath = os.path.join(projectDir, "thirdparty/humanpose/")
thirdpartyBaseLineLibLibPath = os.path.join(projectDir, "thirdparty/human-pose-estimation.pytorch/lib")
thirdpartyYoloPath = os.path.join(projectDir, "thirdparty/yolov3/")

if not DirectoryHelper.DirExists(thirdpartyBaseLinePath) or not DirectoryHelper.DirExists(thirdpartyYoloPath):
    raise Exception("Can't find thirdparty component(s) -> git submodule update --init --recursive")

sys.path.append(thirdpartyYoloPath)
sys.path.append(thirdpartyBaseLinePath)
sys.path.append(thirdpartyBaseLineLibLibPath)
print(sys.path)

pretrainedModelPath = os.path.join(projectDir, "ropose/TrainedModels")

evalPath = os.path.join(projectDir, "../evaluation/")
DirectoryHelper.CreateIfNotExist(evalPath)

showExamplePath = os.path.join(projectDir, "../examples/")
DirectoryHelper.CreateIfNotExist(showExamplePath)

weightsDir = os.path.join(projectDir, "../models/")
DirectoryHelper.CreateIfNotExist(weightsDir)

outputDir = os.path.join(projectDir, "../output/")
DirectoryHelper.CreateIfNotExist(outputDir)

trainedModelBasePath = os.path.join(projectDir, "../trained/")
dataPath = trainedModelBasePath
DirectoryHelper.CreateIfNotExist(trainedModelBasePath)

roposeDatasetDir = os.path.join(projectDir, "../datasets/")
DirectoryHelper.CreateIfNotExist(roposeDatasetDir)

realDataPath = os.path.join(roposeDatasetDir, "real_train/")
roposeEvalDataPath = os.path.join(roposeDatasetDir, "colropose_eval/")
roposeTestDataPath = os.path.join(roposeDatasetDir, "real_test/")

roPoseNetWeights = os.path.join(weightsDir, "ropose_net.pt")
roPoseYoloWeights = os.path.join(weightsDir, "ropose_yolo.pt")
originalHumanPoseModelPath = os.path.join(weightsDir, "pose_resnet_152_256x192.pth.tar")

roposeFineTuneDatasets = [os.path.join(roposeEvalDataPath, "colropose_eval_006")]

simDataPath = os.path.join(roposeDatasetDir, "sim/")

#cocoStuff
cocoPath = homedir + "/coco/"
cocoDatasetTrain = ["train2017"]
cocoDatasetEval = ["val2017"]

# Pose model stuff
linkOrder = ["base_link",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link"]

ropPoseColors = ColorHelper.GetUniqueColors(8)
ropPoseColorsGT = [[0, 0, 255]] * 7

humanColors = ColorHelper.GetUniqueColors(17)
humanColorsGT = [[0, 255, 0]] * 17

humanPairMap = [{0: 1},  #nose - left_eye
                {1: 3},  #left_eye - left_ear
                {0: 2},  #nose - right_eye
                {2: 4},  #right_eye - right_ear
                {3: 5},  #left_ear - left_shoulder
                {4: 6},  #right_ear - right_shoulder

                {6: 8},  # right_shoulder - right_elbow
                {8: 10},  # right_elbow - right_wrist

                {5: 7},  # left_shoulder - left_elbow
                {7: 9},  # left_elbow - left_wrist

                {6: 12},  # right_shoulder - right_hip
                {5: 11},  # left_shoulder - left_hip

                {12: 14},  # right_hip - right_knee
                {14: 16},  # right_knee - right_ankle

                {11: 13},  # left_hip - left_knee
                {13: 15},  # left_knee - left_ankle

                {5: 6},  # left_shoulder - right_shoulder
                {11: 12},  # left_hip - right_hip
                ]

#common net stuff
useHSVColorspace = False
useSpatialDistributionLoss = False
spatialDistributionLossTH = 0.75
# mseHeatmapLoss, spatialDistributionLoss
lossWeights = (1.0, 1.0)
sdlEpochs = 25
advancedFeatureStackSize = 128
rawInputRes = [720, 1280]
inputRes = [256, 192]
outputRes = [64, 48]
maxDistance = math.sqrt(math.pow(outputRes[0], 2) + math.pow(outputRes[1], 2))
paddingValue = 0.0
downsampleFactor = inputRes[0]/outputRes[0]

includeUpsampling = False
if includeUpsampling:
    outputRes = inputRes

loadGrayscale = False
ropose_detectionLayers = linkOrder.__len__() + 1 #XJoints + background
#human pose detection stuff
human_detectionLayers = 17 + 1 #XJoints + background

#collropoe stuff
colropose_freezeFeatureModelEpoch = -1 #20
colropose_freezeRoposeModelEpoch = -1 #100

#ropose CASE model stuff
trainFeatureModel = True
trainBootstrapStage = False
useJointConnectionModel = False
trainJointConnectionModel = True
jointConnectionFilterSize = 64
refineStageCount = 2
jointConnectionStageCount = 1

# define the final comtypes float16 vs. float32 etc.
compType = torch.float16
compTypeNP = np.float16

#training stuff
epochs = 150

#simulation vs real data
mixWithZeroHumans = False
mixWithZeroHuamnsFactor = 0.1
mixRealWithSimulation = False
mixSimulationFactor = 0.25

#Random erasing
useRandomErasing = False
randomErasingProb = 0.10
randomErasingMaxObjectCount = 2
randomErasingMaxAreaCover = 0.10

startLearningRate = 1e-3
changeLearningRate = True
learningRateEpochDecay = 0.005
batchSize = 1
testBatchSize = batchSize
gpuList = ['/gpu:0']

#final prediction config
roposeRejectionTH = 0.70
humanRejectionTH = 0.90

#augmentation
onTheFlyBackgroundAugmentation = False
onTheFlyForegroundAugmentation = False
forgroundAugmentationMaxCount = 5
onTheFlyForegroundAugmentationProb = 0.15
onTheFlyAugmentation = True
onTheFlyAugmentationProbability = 0.7

#ground truth config
if includeUpsampling:
    gaussSigma = 7.0
else:
    gaussSigma = 2.0

pafWidth = 3
heatmapMaxValue = 1.0
heatmapMinValue = 0.0

#input config
keepAspectRatio = True
cropInputToBB = True

#3d joint inputs
use3DJointInput = False
robotMaxWorkSpace = 2.0

#yolo stuff
yolo_BatchSize = 1
yolo_Epochs = 100
yolo_onTheFlyAugmentation = True
yolo_InputSize = [416, 416]# has to be multiple of 32 (Grid size of the used yolo input)
#yolo_InputSize = [320, 320]
#yolo_InputSize = [224, 224] #TinyYolo
#yolo_InputSize = [160, 160]
yolo_StartLearningRate = 0.001
yolo_IgnoreThreshold = 0.5 #conf_thres
yolo_NonMaxSurThreshold = 0.1 #iou_thres

#conf_thres=0.001
#iou_thres=0.6,  # for nms

yolo_Anchors = [[10, 13],  [16, 30],  [33, 23],  [30, 61],  [62, 45],  [59, 119],  [116, 90],  [156, 198],  [373, 326]]

coco_classes = {0: {'supercategory': 'unknown', 'id': 0, 'name': 'unknown'},
                1: {'supercategory': 'person', 'id': 1, 'name': 'person'},
                2: {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
                3: {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
                4: {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
                5: {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
                6: {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
                7: {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
                8: {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
                9: {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
                10: {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
                11: {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
                12: {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
                13: {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
                14: {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'},
                15: {'supercategory': 'animal', 'id': 16, 'name': 'bird'},
                16: {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
                17: {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
                18: {'supercategory': 'animal', 'id': 19, 'name': 'horse'},
                19: {'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
                20: {'supercategory': 'animal', 'id': 21, 'name': 'cow'},
                21: {'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
                22: {'supercategory': 'animal', 'id': 23, 'name': 'bear'},
                23: {'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
                24: {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
                25: {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
                26: {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
                27: {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
                28: {'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
                29: {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
                30: {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'},
                31: {'supercategory': 'sports', 'id': 35, 'name': 'skis'},
                32: {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
                33: {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
                34: {'supercategory': 'sports', 'id': 38, 'name': 'kite'},
                35: {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
                36: {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
                37: {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
                38: {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
                39: {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
                40: {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
                41: {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
                42: {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'},
                43: {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
                44: {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'},
                45: {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
                46: {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'},
                47: {'supercategory': 'food', 'id': 52, 'name': 'banana'},
                48: {'supercategory': 'food', 'id': 53, 'name': 'apple'},
                49: {'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
                50: {'supercategory': 'food', 'id': 55, 'name': 'orange'},
                51: {'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
                52: {'supercategory': 'food', 'id': 57, 'name': 'carrot'},
                53: {'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
                54: {'supercategory': 'food', 'id': 59, 'name': 'pizza'},
                55: {'supercategory': 'food', 'id': 60, 'name': 'donut'},
                56: {'supercategory': 'food', 'id': 61, 'name': 'cake'},
                57: {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
                58: {'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
                59: {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
                60: {'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
                61: {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
                62: {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
                63: {'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
                64: {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
                65: {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
                66: {'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
                67: {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
                68: {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
                69: {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
                70: {'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
                71: {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
                72: {'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
                73: {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
                74: {'supercategory': 'indoor', 'id': 84, 'name': 'book'},
                75: {'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
                76: {'supercategory': 'indoor', 'id': 86, 'name': 'vase'},
                77: {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
                78: {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
                79: {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
                80: {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'},
                81: {'supercategory': 'robot', 'id': 100, 'name': 'ropose robot'}}

yolo_cocoClassMap = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                9: 8,
                10: 9,
                11: 10,
                13: 11,
                14: 12,
                15: 13,
                16: 14,
                17: 15,
                18: 16,
                19: 17,
                20: 18,
                21: 19,
                22: 20,
                23: 21,
                24: 22,
                25: 23,
                27: 24,
                28: 25,
                31: 26,
                32: 27,
                33: 28,
                34: 29,
                35: 30,
                36: 31,
                37: 32,
                38: 33,
                39: 34,
                40: 35,
                41: 36,
                42: 37,
                43: 38,
                44: 39,
                46: 40,
                47: 41,
                48: 42,
                49: 43,
                50: 44,
                51: 45,
                52: 46,
                53: 47,
                54: 48,
                55: 49,
                56: 50,
                57: 51,
                58: 52,
                59: 53,
                60: 54,
                61: 55,
                62: 56,
                63: 57,
                64: 58,
                65: 59,
                67: 60,
                70: 61,
                72: 62,
                73: 63,
                74: 64,
                75: 65,
                76: 66,
                77: 67,
                78: 68,
                79: 69,
                80: 70,
                81: 71,
                82: 72,
                84: 73,
                85: 74,
                86: 75,
                87: 76,
                88: 77,
                89: 78,
                90: 79,
                100: 80,
}

coco_catIDs = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

tinyYoloClasses = {
    0: {'supercategory': 'unknown', 'id': 0, 'name': 'unknown'},
    1: {'supercategory': 'person', 'id': 1, 'name': 'person'},
    2: {'supercategory': 'robot', 'id': 2, 'name': 'robot'}
}

tinyYoloClassMap = {1: 0,
                    2: 1}

yolo_FromScratchParams = os.path.join(projectDir, "yolo_cfg/hyp.scratch.yaml")
yolo_FinetuneParams = os.path.join(projectDir, "yolo_cfg/hyp.finetune.yaml")
yolo_Classes = tinyYoloClasses
yoloClassMap = tinyYoloClassMap
yolo_RoposeClassNum = 1
yolo_ConfigFilePath = os.path.join(projectDir, "yolo_cfg/yolov3_colropose.cfg")

yolo_HumanClassNum = 0
yolo_MaxBoxCount = 50