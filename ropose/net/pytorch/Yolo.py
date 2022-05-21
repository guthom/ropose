from torch import optim, utils
from typing import Dict, List
from ropose.net.pytorch.Util import Timer
import os
import torch
import yaml
from torch import nn
from torch.cuda import amp
from typing import Optional
from ropose.net.pytorch.Util import Util
import numpy as np
from ropose.net.pytorch.NetBase import NetBase
from ropose.net.Training.Generators.pytorch.Yolo import DataGenerator
from ropose_dataset_tools.DataClasses.DetectionTypes.YoloDetection import YoloDetection
from ropose_dataset_tools.Augmentation.Augmentor import Augmentor
from guthoms_helpers.base_types.Vector2D import Vector2D
from ropose.net.pytorch.TrainingHyperParams import TrainingHyperParams
from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D
import ropose.pytorch_config as config
from typing import List, Tuple
from copy import copy, deepcopy
from ropose.net.Training.logging.pytorch.YoloTensorboardLogger import TensorboardLogger
from ropose.net.Training.logging.pytorch.CustomLogger import CustomLogger
from skimage.transform import resize

# stuff from ultraletics
from ropose.thirdparty.yolov3.models import *
from ropose.thirdparty.yolov3.utils.datasets import *

class Yolo(NetBase):
    OutputCount = None
    optimizer = None
    bootstrapModel = None
    jointConnectionModel = None
    featureModel = None
    gpuModels = []

    def __init__(self, trainSet: List[type(Dataset)] = None, testSet: List[type(Dataset)] = None,
                 validationSet: List[type(Dataset)] = None, netResolution: List = None):
        super().__init__(trainSet, testSet, validationSet, netResolution, "Yolo")

        self.learningRate = config.yolo_StartLearningRate

        if testSet is not None:
            self.tensorboardLogger = TensorboardLogger(self.tensorBoardLogPath)
            self.logger = CustomLogger(snapshootDir=os.path.join(self.basePath, "snapshots"),
                                       resultDir=os.path.join(self.basePath, "results"),
                                       modelPath=os.path.join(self.basePath, "CompleteModel.pt"),
                                       model=self, datasets=None, period=1)
        self.augmentor = Augmentor()
        self.classes = config.yolo_Classes
        self.classCount = len(self.classes)
        self.ignoreThreshold = config.yolo_IgnoreThreshold
        self.exampleCounter = 0

        # stuff from ultraletics adapted from https://github.com/ultralytics/yolov5/blob/master/detect.py
        self.imgsz: Optional[int] = None
        self.stride: Optional[int] = None

    def LoadHyperParams(self, path: str) -> Dict:
        ret = None
        with open(path) as file:
            ret = yaml.load(file, Loader=yaml.FullLoader)
        return ret

    def DefineModel(self):

        # hyperparameters from ultralytics -> https://github.com/ultralytics/yolov5/blob/master/train.py
        # we will take the fromsratch parameters here https://github.com/ultralytics/yolov5/blob/6f5d6fcdaa8c1c5b24a06fdf9fd4e12c781fb4f7/data/hyp.scratch.yaml#L1-L33
        self.yoloHyp = self.LoadHyperParams(config.yolo_FromScratchParams)
        # it looks like the origin author forgot to add this to the above config files
        self.yoloHyp['giou'] = 3.54,
        self.nc = 2

        self.netModel = Darknet(config.yolo_ConfigFilePath).to(self.device)
        self.hyp = self.yoloHyp

        return


    def LoadPretrainedModel(self, path: str = None):
        try:
            if path is not None:
                if os.path.isfile(path):
                    # Load model
                    temp = torch.load(path, map_location=self.device)
                    self.netModel.load_state_dict(temp['model'])
                    return
                else:
                    raise Exception("Custom path is no valid file!")
        except:
            if os.path.isfile(path):
                self.netModel.load_state_dict(torch.load(path))
                self.netModel.eval()
                print("Loaded selftrained YOLO Model!")
                return

        path = os.path.join(self.basePath, "CompleteModel.pt")
        if os.path.isfile(path):
            self.netModel.load_state_dict(torch.load(path))
            self.netModel.eval()
            print("Loaded selftrained YOLO Model!")
        raise Exception("Original Model File does not exist and no path was specified!")

    def PrepareInput(self, image: np.array, detections: List[YoloDetection], size=config.yolo_InputSize) -> np.array:
        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))

        if config.keepAspectRatio:
            # Add padding
            image = np.pad(image, pad, 'constant', constant_values=255 * config.paddingValue)

        resizedImage = resize(image, (size[0], size[1], 3), mode='reflect', anti_aliasing=False)

        fv = size[0] / image.shape[0]
        fu = size[1] / image.shape[1]

        resizedDetections = []
        for detection in detections:
            resizedBB = detection.boundingBox
            resizedBB = resizedBB.AddPadding(pad[1][0], pad[0][0])
            resizedBB = resizedBB.ScaleCoordiantes(Vector2D(fv, fu))
            detection.boundingBox = resizedBB
            resizedDetections.append(detection)

        return resizedImage, resizedDetections

    def FillLabels(self, labels: List[np.array]):
        filled_labels = []
        filled_labels.extend(labels)

        if filled_labels.__len__() < config.yolo_MaxBoxCount:
            fill = config.yolo_MaxBoxCount - filled_labels.__len__()
            for i in range(0, fill):
                filled_labels.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        elif filled_labels.__len__() >= config.yolo_MaxBoxCount:
            filled_labels = filled_labels[:config.yolo_MaxBoxCount]

        return np.array(filled_labels)

    def SaveFineTunedExamples(self, x, y,  basePath: str = "/mnt/datastuff/TestExamples/yolo_finetune_gt"):
        filepathImage = os.path.join(basePath, "fineTuneExample_" + str(self.exampleCounter) + ".jpg")
        plt = Util.DrawYoloBBGT(x, y)
        plt.savefig(filepathImage)
        plt.close()
        self.exampleCounter += 1

    def ActivateFinetuning(self):
        modules = self.netModel.module_list

        for module in modules:
            if module.__class__.__name__ == 'YOLOLayer':
                module.requires_grad = True
            else:
                module.requires_grad = False


    def FineTuneOnYoloDetection(self, dataPairs: List[Tuple[np.array, List[YoloDetection]]],
                                hyperParams: TrainingHyperParams, augmentation: bool = True,
                                augmentationAmount: int = 10, saveExamples: bool=False):
        #self.ActivateFinetuning()
        self.yoloHyp = self.LoadHyperParams(config.yolo_FinetuneParams)
        self.netModel.hyp = self.yoloHyp
        self.netModel.nc = self.nc
        # taken form the repos training routine
        # https://github.com/ultralytics/yolov3/blob/master/train.py
        self.netModel.gr = 1.0

        optimizer = hyperParams.GetOptimizer(self.netModel)

        batchList = list()
        for image, detections in dataPairs:

            x, detections = self.PrepareInput(image, detections)

            if augmentation:
                for i in range(0, augmentationAmount):
                    myX = deepcopy(x)
                    myDetectections = deepcopy(detections)
                    bbs = []
                    for detection in myDetectections:
                        bbs.append(detection.boundingBox.ToIaaBoundingBox())

                    myX, augmentedY = self.augmentor.AugmentImagesAndBBs(myX, bbs, forceAugmentation=True)

                    augY = []
                    for j in range(0, len(myDetectections)):
                        myDetectections[j].boundingBox = BoundingBox2D.FromIaaBB(augmentedY[j])
                        augY.append(myDetectections[j].ToPredictionTensor(image=myX))

                    if saveExamples:
                        self.SaveFineTunedExamples(myX, augY)

                    yTorch = torch.from_numpy(np.array(augY)).float()

                    myX = myX.transpose(2, 0, 1)
                    myX = torch.from_numpy(myX).float()

                    batchList.append([myX, yTorch])
            else:
                #add untouched example
                y = []
                for example in detections:
                    myExample = deepcopy(example)
                    y.append(myExample.ToPredictionTensor(image=x))

                if saveExamples:
                    self.SaveFineTunedExamples(x, y)

                y = torch.from_numpy(np.array(y))
                x = x.transpose(2, 0, 1)
                x = torch.from_numpy(x).float()
                # x = (x.unsqueeze(dim=0))
                batchList.append([x, y])

        self.netModel.train()
        random.shuffle(batchList)

        for i in range(0, batchList.__len__(), hyperParams.batchSize):
            batch = tuple(batchList[i:i+hyperParams.batchSize])
            x, y = DataGenerator.collate_fn(batch)
            y = y.to(self.device).float()
            x = x.to(self.device).float()

            optimizer.zero_grad()
            pred = self.netModel(x)
            loss, lossItems = compute_loss(pred, y, self.netModel)
            loss.backward()
            optimizer.step()
            print("Retrained Yolo with loss: " + str(loss.item()))
        #back to eval mode
        self.netModel.eval()

    def TrainModel(self, hyperParams: TrainingHyperParams, trainSet=None, testSet=None):
        raise Exception("Not Implemented! - Please use the DataGenerator and take the training routine from the original repo!")

    def Predict(self, inp: torch.Tensor):
        self.netModel.eval()
        _inp = self.PreprocessInput(inp)
        with torch.no_grad():
            input = _inp.to(self.device)
            ret = self.netModel(input)
        return ret[0]

    def PreprocessInput(self, inp: torch.Tensor):
        #return preprocess_input(input, mode='tf')
        return inp
