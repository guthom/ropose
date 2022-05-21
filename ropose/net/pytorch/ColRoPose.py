import numpy as np
import torch
import random
import os
from torch import nn, optim, utils
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from ropose.net.pytorch.Util import Util, Timer

from ropose.net.pytorch.NetBase import NetBase
from ropose.net.Training.Generators.pytorch.ColRopose import DataGenerator
from ropose.net.pytorch.TrainingHyperParams import TrainingHyperParams

import ropose.pytorch_config as config
from typing import List, Dict
from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset

import matplotlib.pyplot as plt
from ropose.net.Training.logging.pytorch.ColRoposeTensorboardLogger import TensorboardLogger
from ropose.net.Training.logging.pytorch.ColRoposeCustomLogger import CustomLogger
from ropose.net.pytorch.PartModells import ResNetC5, Human_SimpleBaseline, RoPose_SimpleBaseline
from ropose.net.Training.Losses.pytorch.MultipleHeatmapLoss import MultipleHeatmapLoss

class ColRoPose(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resNet = None
        self.poseFeatureModel = None
        self.ropose = None
        self.humanpose = None
        self.DefineModel()

    def forward(self, x):
        resNet = self.resNet(x)

        roposeOut = self.ropose(resNet)
        humanPoseOut = self.humanpose(resNet)

        return roposeOut, humanPoseOut

    def DefineModel(self):
        self.resNet = ResNetC5()
        self.resropose.net.to(self.device)

        self.ropose = RoPose_SimpleBaseline()
        self.ropose.to(self.device)

        self.humanpose = Human_SimpleBaseline()
        self.humanpose.to(self.device)


class ColRoPoseNet(NetBase):
    OutputCount = None
    optimizer = None
    bootstrapModel = None
    jointConnectionModel = None
    featureModel = None
    gpuModels = []

    def __init__(self, trainSet: List[type(Dataset)] = None, testSet: List[type(Dataset)] = None,
                 validationSet: List[type(Dataset)] = None, netResolution: List = None):
        super().__init__(trainSet, testSet, validationSet, netResolution, "ColRopose")

    def DefineModel(self):
        self.netModel = ColRoPose()
        self.netModel.to(self.device)
        return


    def TrainModel(self, roposeTrain=None, roposeTest=None, humanTrain=None, humanTest=None, batch_size=32, epochs=1):
        self.tensorboardLogger = TensorboardLogger(self.tensorBoardLogPath, self, [roposeTest[0], humanTest[0]])
        self.logger = CustomLogger(snapshootDir=os.path.join(self.basePath, "snapshots"),
                                     resultDir=os.path.join(self.basePath, "results"),
                                     modelPath=os.path.join(self.basePath, "CompleteModel.pt"),
                                     model=self, datasets=[roposeTest[0], humanTest[0]], period=1)

        dataDataset = DataGenerator(roPoseDatasets=roposeTrain, humanDatasets=humanTrain, path=config.cocoDataset,
                                    preprocessFunction=self.PreprocessInput)
        dataLoader = utils.data.DataLoader(dataDataset, batch_size=batch_size, shuffle=True, num_workers=16)
        steps = len(dataLoader)

        testDataset = DataGenerator(roPoseDatasets=roposeTest, humanDatasets=humanTest, path=config.cocoDataset,
                                    preprocessFunction=self.PreprocessInput)
        testLoader = utils.data.DataLoader(testDataset, batch_size=config.testBatchSize, shuffle=True, num_workers=16)

        lossFunction = MultipleHeatmapLoss().to(self.device)

        #learning and LR-Scheduling
        #optimizer = optim.Adam(self.netModel.parameters(), lr=self.learningRate)
        #optimizer = optim.SGD(self.netModel.parameters(), lr=self.learningRate, momentum=0.9)

        #Base LR  1e-3. It drops to 1e-4 at 90 epochs and 1e-5 at 120 epochs.
        #See https://arxiv.org/pdf/1804.06208.pdf
        #lrScheduler = MultiStepLR(optimizer, [50, 90, 110], 0.1)

        lrScheduler = ExponentialLR(optimizer, gamma=0.99)

        timer = Timer()
        zeroHuman = [Util.CreateZeroHeatmaps(config.human_detectionLayers)]
        zeroRopos = [Util.CreateZeroHeatmaps(config.ropose_detectionLayers)]

        for epoch in range(epochs):
            self.netModel.train()
            lrScheduler.step()
            dataDataset.ShuffleHumanSets()

            if config.colropose_freezeFeatureModelEpoch != -1 and epoch == config.colropose_freezeFeatureModelEpoch:
                print("Freeze Feature Layers!")
                self.FreezeModel(self.netModel.resNet)

            if config.colropose_freezeRoposeModelEpoch != -1 and epoch == config.colropose_freezeRoposeModelEpoch:
                print("Freeze RoPose Layers!")
                self.FreezeModel(self.netModel.ropose)

            for i, (inp_1, gt_1, inp_2, gt_2) in enumerate(dataLoader):
                timer.Start()

                '''
                img = gt.numpy()[0, :, :, :]
                imgplot = plt.imshow(img[7, :, :])
                plt.show()

                img = gt.numpy()[0, :, :, :]
                imgplot = plt.imshow(img[1, :, :])
                plt.show()

                img = inp.numpy()[0, :, :, :]

                img = img.transpose((1, 2, 0))
                imgplot = plt.imshow(img)
                plt.show()
                '''
                currentBatch = inp_1.shape[0]
                zeroHumans = torch.stack(zeroHuman * currentBatch).to(self.device)
                zeroRopose = torch.stack(zeroRopos * currentBatch).to(self.device)

                inp_1 = inp_1.cuda().float()
                gt_1 = gt_1.cuda().float()
                inp_2 = inp_2.cuda().float()
                gt_2 = gt_2.cuda().float()

                #x = self.preprocessTransform(inp_1)
                outputs = self.netModel(inp_1)
                #add zero human maps for this run
                optimizer.zero_grad()
                loss_1 = lossFunction(outputs[0], gt_1)
                loss_1 += lossFunction(outputs[1], zeroHumans)
                loss_1.backward()
                optimizer.step()

                #x = self.preprocessTransform(inp_2)
                outputs = self.netModel(inp_2)
                #add zero ropose maps for this run
                optimizer.zero_grad()
                loss_2 = lossFunction(outputs[0], zeroRopose)
                loss_2 += lossFunction(outputs[1], gt_2)
                loss_2.backward()
                optimizer.step()

                time = timer.Stop(printLine=False)
                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss_1: {:.4f}, Time: {}'
                          .format(epoch + 1, epochs, i + 1, steps, loss_1.item(), time))
                    print('Epoch [{}/{}], Step [{}/{}], Loss_2: {:.4f}, Time: {}'
                          .format(epoch + 1, epochs, i + 1, steps, loss_2.item(), time))

            self.netModel.eval()

            losses_1 = []
            losses_2 = []

            with torch.no_grad():
                for inp_1, gt_1, inp_2, gt_2 in testLoader:
                    inp_1 = inp_1.cuda().float()
                    gt_1 = gt_1.cuda().float()
                    inp_2 = inp_2.cuda().float()
                    gt_2 = gt_2.cuda().float()

                    currentBatch = inp_1.shape[0]
                    zeroHumans = torch.stack(zeroHuman * currentBatch).to(self.device)
                    zeroRopose = torch.stack(zeroRopos * currentBatch).to(self.device)

                    outputs = self.netModel(inp_1)
                    loss_1 = lossFunction(outputs[0], gt_1)
                    loss_1 += lossFunction(outputs[1], zeroHumans)
                    losses_1.append(loss_1.item())
                    outputs = self.netModel(inp_2)
                    loss_2 = lossFunction(outputs[0], zeroRopose)
                    loss_2 += lossFunction(outputs[1], gt_2)
                    losses_2.append(loss_2.item())

                meanLoss_1 = np.mean(np.array(losses_1))
                meanLoss_2 = np.mean(np.array(losses_2))
                print('MeanTestLoss_1: ' + str(meanLoss_1))
                print('MeanTestLoss_2: ' + str(meanLoss_2))
                self.tensorboardLogger.LogEpoch(meanLoss=[meanLoss_1, meanLoss_2], epoch=epoch,
                                                lr=lrScheduler.get_lr()[0])
                self.logger.EpochEnd(epoch)


    def Predict(self, input: torch.Tensor):
        with torch.no_grad():
            ret = self.netModel(input)
        return ret

    def PreprocessInput(self, inp: torch.Tensor):
        return self.preprocessTransform(inp)
