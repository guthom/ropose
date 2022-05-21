import torch
import torch.nn as nn
import torchvision.transforms as transforms
from abc import abstractmethod
import os
from typing import List, Dict
import abc
import ropose.pytorch_config as config
from shutil import copyfile
from ropose.net.pytorch.TrainingHyperParams import TrainingHyperParams
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
from torchsummary import summary
import numpy as np
from ropose.net.Training.logging.pytorch.TensorboardLogger import TensorboardLogger
from ropose.net.Training.logging.pytorch.CustomLogger import CustomLogger
import json

class PartModel(nn.Module):

    def __init__(self):
        self.layers: List['nn.Sequential']

        super().__init__()
        self.netModel = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cpu = "cpu"
        self.layers = []
        self.outputLayer = None
        self.DefineModel()

    def MakeModel(self):
        self.netModel = nn.Sequential(*list(self.layers)).to(self.device)

    @abstractmethod
    def DefineModel(self):
        pass

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        if self.outputLayer is not None:
            return self.outputLayer(out)
        else:
            return out

class NetBase(object):
    __metaclass__ = abc

    modelName = "Base"
    device = None
    netModel = None
    gpuModel = None
    inputShape = None
    netResolution = config.inputRes
    preprocessTransform = None
    trainSet = None
    testSet = None
    validationSet = None

    def __init__(self, trainSet=None, testSet=None, validationSet=None, netResolution: List=None, modelName: str="Base"):

        self.preprocessTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.modelName = modelName
        print("Init Model: " + modelName)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cpu = "cpu"
        self.basePath = os.path.join(config.pretrainedModelPath, "pytorch", self.modelName)

        if testSet is not None:
            self.logger = CustomLogger(snapshootDir=os.path.join(self.basePath, "snapshots"),
                                       resultDir=os.path.join(self.basePath, "results"),
                                       modelPath=os.path.join(self.basePath, "CompleteModel.pt"),
                                       model=self, datasets=[testSet[0]], period=1)
        else:
            self.logger = CustomLogger(snapshootDir=os.path.join(self.basePath, "snapshots"),
                                       resultDir=os.path.join(self.basePath, "results"),
                                       modelPath=os.path.join(self.basePath, "CompleteModel.pt"),
                                       model=self, datasets=[], period=1)

        self.tensorBoardLogPath = os.path.join(self.basePath, "logs")

        if netResolution is not None:
            self.netResolution = netResolution

        self.CheckDirs()

        self.inputShape = (3, self.netResolution[0], self.netResolution[1])

        self.DefineModel()

        #summary(self.netModel, self.inputShape)

        self.trainSet = trainSet
        self.testSet = testSet
        self.validationSet = validationSet

        return

    def InitModelWeigths(self):
        pass

    def ShareMemory(self):
        self.netModel.share_memory()

    def FreezeModel(self, model):
        for child in model.children():
            for param in child:
                param.requires_grad = False

    def CheckDirs(self):

        if not os.path.isdir(self.basePath):
            os.makedirs(self.basePath, exist_ok=True)

        path = os.path.join(self.basePath, "results")
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        path = os.path.join(self.basePath, "logs")
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        path = os.path.join(self.basePath, "snapshots")
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    @abc.abstractclassmethod
    def PreprocessInput(self, inp):
        raise Exception("Not Implemented!")

    def SaveModelWeights(self, path: str=None):
        if path is not None:
            filePath = path
        else:
            filePath = os.path.join(self.basePath, "CompleteModel.pt")

        torch.save(self.netModel.state_dict(), filePath)

    def CopyCompleteModel(self, path):
        completeModelPath = os.path.join(self.basePath, "CompleteModel.pt")
        copyfile(completeModelPath, path)

    def LoadPretrainedModel(self, path: str= None):

        if path is not None:
            filePath = path
        else:
            filePath = os.path.join(self.basePath, "CompleteModel.pt")

        if os.path.isfile(filePath):
            self.netModel.load_state_dict(torch.load(filePath))
            self.netModel.to(self.device)
            #self.netModel.eval()
            print("Loaded Weights to Model - " + self.modelName + "!")
        else:
            raise Exception("Model File does not exist!")
        return

    def FinishTraining(self, hyperParams: TrainingHyperParams, bestLoss: float = None):
        #check if path exist, create it otherwise
        DirectoryHelper.CreateIfNotExist(hyperParams.finalModelDir)

        finalModelPath = os.path.join(hyperParams.finalModelDir, hyperParams.GetFinalModelName())

        self.CopyCompleteModel(finalModelPath)
        print("Saved best Model to" + finalModelPath)
        hyperParams.SaveSummary(bestLoss=bestLoss)
        print("Saved hyperparams to" + finalModelPath)

    @abc.abstractclassmethod
    def Predict(self, input):
        input = self.PreprocessInput(inp=input)
        return self.netModel(input)

    @abc.abstractclassmethod
    def TrainModel(self, hyperParams: TrainingHyperParams, trainSet=None, testSet=None):
        raise NotImplementedError("Please Implement this method")

    @abc.abstractmethod
    def FineTuneOnBatch(self, batch, hyperParams: TrainingHyperParams):
        raise NotImplementedError("Please Implement this method")

    @abc.abstractclassmethod
    def DefineModel(self):
        pass

    @abc.abstractclassmethod
    def Evaluate(self):
        raise NotImplementedError("Please Implement this method")

