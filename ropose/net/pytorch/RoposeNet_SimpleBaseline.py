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
from ropose.net.pytorch.PartModells import ResNetC5, RoPose_SimpleBaseline
from ropose.net.Training.Losses.pytorch.MultipleHeatmapLoss import MultipleHeatmapLoss
from ropose.net.Training.Losses.pytorch.SpatialDistributionLoss import SpatialDistributionLoss
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
        self.simpleBaseline = None
        self.DefineModel()

    def forward(self, x):
        out = self.resNet(x)

        if config.use3DJointInput:
            out = self.poseFeatureModel(out)

        out = self.simpleBaseline(out)

        return out

    def DefineModel(self):
        self.resNet = ResNetC5()
        self.resNet.to(self.device)

        #self.poseFeatures = RoPoseFeatureExtractor()
        self.simpleBaseline = RoPose_SimpleBaseline()
        self.simpleBaseline.to(self.device)


class RoposNet(NetBase):
    OutputCount = None
    optimizer = None
    bootstrapModel = None
    jointConnectionModel = None
    featureModel = None
    gpuModels = []

    def __init__(self, trainSet: List[type(Dataset)] = None, testSet: List[type(Dataset)] = None,
                 validationSet: List[type(Dataset)] = None, netResolution: List = None, name="RoPose_SimpleBaseLine"):
        super().__init__(trainSet, testSet, validationSet, netResolution, name)

        if testSet is not None:
            self.tensorboardLogger = TensorboardLogger(self.tensorBoardLogPath, self, testSet[0])


    def DefineModel(self):
        self.netModel = RoPoseModel()
        self.InitModelWeigths()
        self.netModel.to(self.device)
        return

    def TrainModel(self, hyperParams: TrainingHyperParams, trainSet=None, testSet=None):
        print("Will train " + self.modelName + " with " + str(len(trainSet)) + " Datasets and " +
              str(len(testSet)) + " Testsets.")
        trainDataset = DataGenerator(datasets=trainSet, trainingParametrs=hyperParams,
                                     preprocessFunction=self.PreprocessInput, augmentation=True)
        trainLoader = utils.data.DataLoader(trainDataset, batch_size=hyperParams.batchSize, shuffle=True,
                                            num_workers=12)
        steps = len(trainLoader)

        testDataset = DataGenerator(datasets=testSet, trainingParametrs=hyperParams, augmentation=False,
                                    preprocessFunction=self.PreprocessInput)

        testLoader = utils.data.DataLoader(testDataset, batch_size=hyperParams.batchSize, shuffle=True,
                                           num_workers=12)
        lossFuctions = []
        if config.useSpatialDistributionLoss:
            heaptmapLossFunction = MultipleHeatmapLoss(weight=config.lossWeights[0]).to(self.device)
            lossFuctions.append(heaptmapLossFunction)
            distributionLossFunction = SpatialDistributionLoss(maxDistance=config.maxDistance,
                                                               weight=config.lossWeights[1]).to(self.device)
            lossFuctions.append(distributionLossFunction)
        else:
            heaptmapLossFunction = MultipleHeatmapLoss(weight=1.0).to(self.device)
            lossFuctions.append(heaptmapLossFunction)

        optimizer = hyperParams.GetOptimizer(self.netModel)

        lrScheduler = hyperParams.GetScheduler(optimizer)

        timer = Timer()
        epochs = hyperParams.epochs

        for epoch in range(epochs):
            self.netModel.train()
            if lrScheduler is not None:
                lrScheduler.step()
                print("Learning rate have been set to: " + str(lrScheduler.get_lr()[0]))
            sdlActive = config.useSpatialDistributionLoss and not epoch >= config.sdlEpochs

            if config.useSpatialDistributionLoss and epoch == config.sdlEpochs:
                # lets use the normal loss after the initial phase with sdl
                lossFuctions = [MultipleHeatmapLoss(weight=1.0).to(self.device)]

            for i, (inp, gt) in enumerate(trainLoader):
                timer.Start()

                '''
                img = gt.numpy()[0, :, :, :]
                imgplot = plt.imshow(img[7, :, :])

                img = gt.numpy()[0, :, :, :]
                map = img[0, :, :]
                for j in range(1, 7):
                    map += img[j, :, :]

                imgplot = plt.imshow(map)
                plt.savefig('/mnt/datastuff/TestExamples/ropose/ropose_' + str(i) + '_0.jpg')

                img = inp.numpy()[0, :, :, :]
                img = img.transpose((1, 2, 0))
                imgplot = plt.imshow(img)
                plt.savefig('/mnt/datastuff/TestExamples/ropose/ropose_' + str(i) + '_1.jpg')
                '''

                inp = inp.to(self.device)
                gt = gt.to(self.device).float()

                optimizer.zero_grad()

                outputs = self.netModel(inp)

                losses = []
                lossValues = []
                for _loss in lossFuctions:
                    losses.append(_loss(outputs, gt))
                    lossValues.append(losses[-1].item())

                summedLoss = torch.stack(losses).sum()

                summedLoss.backward()
                optimizer.step()

                time = timer.Stop(printLine=False)
                if (i + 1) % 1 == 0:
                    if sdlActive:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}, PartLosses: {:.6f}, {:.6f} ,Time: {}'
                              .format(epoch + 1, epochs, i + 1, steps, summedLoss.item(), lossValues[0], lossValues[1],
                                      time))
                    else:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f},Time: {}'
                              .format(epoch + 1, epochs, i + 1, steps, summedLoss.item(), time))

            #self.netModel.eval()

            computedLosses = []
            computetPartLosses = []
            with torch.no_grad():
                for inp, gt in testLoader:
                    inp = inp.to(self.device)
                    gt = gt.to(self.device).float()
                    outputs = self.netModel(inp)

                    losses = []
                    lossValues = []
                    for _loss in lossFuctions:
                        losses.append(_loss(outputs, gt))
                        lossValues.append(losses[-1].item())

                    summedLoss = torch.stack(losses).sum()
                    computedLosses.append(summedLoss.item())
                    computetPartLosses.append(lossValues)

                meanLoss = np.mean(computedLosses)
                meanPartLosses = np.mean(computetPartLosses, axis=0)
                print('MeanTestLoss: ' + str(meanLoss))
                #self.tensorboardLogger.LogEpoch(meanLoss=meanLoss, epoch=epoch, lr=lrScheduler.get_lr()[0])
                if lrScheduler is not None:
                    self.logger.EpochEnd(epoch, meanLoss, meanPartLosses.tolist(), lrScheduler.get_lr())
                else:
                    self.logger.EpochEnd(epoch, meanLoss, meanPartLosses.tolist())

        return self.logger.bestLoss

    def Predict(self, input: torch.Tensor):
        with torch.no_grad():
            input = input.to(self.device)
            ret = self.netModel(input)
        return ret

    def PreprocessInput(self, inp: torch.Tensor):
        return self.preprocessTransform(inp)
