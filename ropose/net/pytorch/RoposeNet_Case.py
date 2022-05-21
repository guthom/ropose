import numpy as np
import os
import torch
from torch import nn, optim, utils
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from ropose.net.pytorch.Util import Timer

from ropose.net.pytorch.NetBase import NetBase
from ropose.net.Training.Generators.pytorch.Ropose_SimpleBaseline import DataGenerator
from ropose.net.pytorch.TrainingHyperParams import TrainingHyperParams
import ropose.pytorch_config as config
from typing import List
from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset
import matplotlib.pyplot as plt
from ropose.net.pytorch.PartModells import Vgg16, RoPose_Case
from ropose.net.Training.Losses.pytorch.MultipleHeatmapLoss import MultipleHeatmapLoss
from ropose.net.Training.logging.pytorch.TensorboardLogger import TensorboardLogger
from ropose.net.Training.logging.pytorch.CustomLogger import CustomLogger

from torchvision.transforms import Normalize, ToPILImage
from ropose.net.pytorch.Util import Util as util
import cv2
import copy


class RoPoseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resNet = None
        self.poseFeatureModel = None
        self.ropose = None
        self.DefineModel()

    def forward(self, x):
        out = self.resNet(x)
        out = self.ropose(out)
        return out

    def DefineModel(self):
        self.resNet = Vgg16()
        self.resNet.to(self.device)

        #self.poseFeatures = RoPoseFeatureExtractor()
        self.ropose = RoPose_Case()
        self.ropose.to(self.device)


class RoposNet(NetBase):
    OutputCount = None
    optimizer = None
    bootstrapModel = None
    jointConnectionModel = None
    featureModel = None
    gpuModels = []

    def __init__(self, trainSet: List[type(Dataset)] = None, testSet: List[type(Dataset)] = None,
                 validationSet: List[type(Dataset)] = None, netResolution: List = None):
        super().__init__(trainSet, testSet, validationSet, netResolution, "RoPose_SimpleBaseLine")

        if testSet is not None:
            self.tensorboardLogger = TensorboardLogger(self.tensorBoardLogPath, self, testSet[0])
            self.logger = CustomLogger(snapshootDir=os.path.join(self.basePath, "snapshots"),
                                       resultDir=os.path.join(self.basePath, "results"),
                                       modelPath=os.path.join(self.basePath, "CompleteModel.pt"),
                                       model=self, datasets=[testSet[0]], period=1)

    def DefineModel(self):
        self.netModel = RoPoseModel()
        self.netModel.to(self.device)
        return

    def TrainModel(self, hyperParams: TrainingHyperParams, trainSet=None, testSet=None):

        trainDataset = DataGenerator(datasets=trainSet, trainingParametrs=hyperParams,
                                     preprocessFunction=self.PreprocessInput)
        trainLoader = utils.data.DataLoader(trainDataset, batch_size=hyperParams.batchSize, shuffle=True,
                                            num_workers=16)
        steps = len(trainLoader)

        testDataset = DataGenerator(datasets=testSet, trainingParametrs=hyperParams,
                                    preprocessFunction=self.PreprocessInput)

        testLoader = utils.data.DataLoader(testDataset, batch_size=hyperParams.batchSize, shuffle=True, num_workers=16)

        lossFunction = MultipleHeatmapLoss().to(self.device)

        optimizer = hyperParams.GetOptimizer(self.netModel)

        lrScheduler = hyperParams.GetScheduler(optimizer)

        timer = Timer()
        epochs = hyperParams.epochs

        for epoch in range(epochs):
            self.netModel.train()
            lrScheduler.step()
            for i, (inp, gt) in enumerate(trainLoader):
                timer.Start()

                '''
                img = gt.numpy()[0, :, :, :]
                imgplot = plt.imshow(img[7, :, :])
                plt.show()

                img = gt.numpy()[0, :, :, :]
                map = img[0, :, :]
                for i in range(1, 7):
                    map += img[i, :, :]

                imgplot = plt.imshow(map)
                plt.show()

                img = inp.numpy()[0, :, :, :]
                img = img.transpose((1, 2, 0))
                imgplot = plt.imshow(img)
                plt.show()
                '''

                inp = inp.to(self.device)
                gt = gt.to(self.device).float()

                optimizer.zero_grad()

                outputs = self.netModel(inp)
                loss = lossFunction(outputs, gt)

                loss.backward()
                optimizer.step()

                time = timer.Stop(printLine=False)
                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {}'
                          .format(epoch + 1, epochs, i + 1, steps, loss.item(), time))

            #self.netModel.eval()

            losses = []
            with torch.no_grad():
                for inp, gt in testLoader:
                    inp = inp.to(self.device)
                    gt = gt.to(self.device).float()
                    outputs = self.netModel(inp)
                    losses.append(lossFunction(outputs, gt).to(self.cpu))

                meanLoss = np.mean(np.array(losses))
                print('MeanTestLoss: ' + str(meanLoss))
                #self.tensorboardLogger.LogEpoch(meanLoss=meanLoss, epoch=epoch, lr=lrScheduler.get_lr()[0])
                self.logger.EpochEnd(epoch, float(meanLoss))

        return self.logger.bestLoss

    def Predict(self, input: torch.Tensor):
        with torch.no_grad():
            input = input.to(self.device)
            ret = self.netModel(input)
        return ret

    def PreprocessInput(self, inp: torch.Tensor):
        return self.preprocessTransform(inp)
