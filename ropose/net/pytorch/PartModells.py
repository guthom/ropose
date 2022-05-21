import numpy as np
import torch
from typing import List
import torch.nn as nn
from torch.nn import Flatten
import torch.utils
from torchvision.models import resnet50, vgg16, vgg19
from ropose.net.pytorch.NetBase import NetBase, PartModel
import ropose.pytorch_config as config

class ResNetC5(PartModel):
    def __init__(self):
        super().__init__()
        self.netModel = resnet50(pretrained=True)
        # get rid of the top layers in resNet start with c5 as described in
        # https://arxiv.org/pdf/1804.06208.pdf
        children = list(self.netModel.children())
        self.netModel = nn.Sequential(*list(self.netModel.children())[:-2]).to(self.device)
        self.layers = list(self.netModel.children())

class Vgg16(PartModel):
    def __init__(self):
        super().__init__()
        self.netModel = vgg16(pretrained=True)
        # get rid of the top layers
        children = list(self.netModel.children())
        raise Exception("Not finally Implemented!")
        self.netModel = nn.Sequential(*list(self.netModel.children())[:-2]).to(self.device)
        self.layers = list(self.netModel.children())

class Vgg19(PartModel):
    def __init__(self):
        super().__init__()
        self.netModel = vgg19(pretrained=True)
        # get rid of the top layers
        children = list(self.netModel.children())
        raise Exception("Not finally Implemented!")
        self.netModel = nn.Sequential(*list(self.netModel.children())[:-2]).to(self.device)
        self.layers = list(self.netModel.children())

class RoPose_Case(PartModel):
    def __init__(self):
        self.batchNormMomentum = 0.1
        super().__init__()

class RoPose_SimpleBaseline(PartModel):
    def __init__(self):
        self.batchNormMomentum = 0.1
        super().__init__()

    def DeConvLayer(self, inChannels: int=256, filterSize: int=256, kernelSize: int=4, stride: int=2, padding: int=1,
                    outputPadding: int=0) -> nn.Sequential:
        self.layers.append(nn.ConvTranspose2d(in_channels=inChannels, out_channels=filterSize,
                                              kernel_size=(kernelSize, kernelSize),
                                              stride=(stride, stride), padding=(padding, padding),
                                              output_padding=(outputPadding, outputPadding), bias=False
                                              ))

        self.layers.append(nn.BatchNorm2d(filterSize, momentum=self.batchNormMomentum))
        self.layers.append(nn.ReLU(True))
        return self.layers[-1]

    def OutputLayer(self, inChannels:int, outputLayers:int):
        self.layers.append(nn.Conv2d(inChannels, outputLayers, kernel_size=(1, 1), stride=(1, 1)))

    def UpsamplingLayer(self, factor: float = config.downsampleFactor):
        self.layers.append(torch.nn.Upsample(scale_factor=factor, mode="bicubic"))

    def DefineModel(self):
        self.DeConvLayer(inChannels=2048)
        self.DeConvLayer()
        self.DeConvLayer()
        self.OutputLayer(inChannels=256, outputLayers=config.ropose_detectionLayers)

        if config.includeUpsampling:
            self.UpsamplingLayer()

        self.netModel = nn.Sequential(*list(self.layers)).to(self.device)

class RoposeRefinementStage(PartModel):

    def __init__(self):
        self.batchNormMomentum = 0.1
        super().__init__()


    def DefineModel(self):
        outputLayers = config.ropose_detectionLayers
        inputChannels = outputLayers + config.advancedFeatureStackSize
        inChannels: int = 256
        filterSize: int = 256
        kernelSize: int = 4
        stride: int = 1
        outputPadding = 0
        padding: int = 0
        extractionDepth = 2

        self.layers.append(nn.Conv2d(in_channels=inputChannels, out_channels=filterSize,
                                     kernel_size=(kernelSize, kernelSize),
                                     stride=(stride, stride), padding=(padding, padding), bias=False))
        self.layers.append(nn.BatchNorm2d(filterSize, momentum=self.batchNormMomentum))
        self.layers.append(nn.ReLU(True))

        for i in range(0, extractionDepth-1):
            self.layers.append(nn.Conv2d(in_channels=inChannels, out_channels=filterSize,
                                 kernel_size=(kernelSize, kernelSize),
                                 stride=(stride, stride), padding=(padding, padding), bias=False))
            self.layers.append(nn.BatchNorm2d(filterSize, momentum=self.batchNormMomentum))
            self.layers.append(nn.ReLU(True))

        for i in range(0, extractionDepth):
            self.layers.append(nn.ConvTranspose2d(in_channels=inChannels, out_channels=filterSize,
                                          kernel_size=(kernelSize, kernelSize),
                                          stride=(stride, stride), padding=(padding, padding),
                                          output_padding=(outputPadding, outputPadding), bias=False))
            self.layers.append(nn.BatchNorm2d(filterSize, momentum=self.batchNormMomentum))
            self.layers.append(nn.ReLU(True))

        self.layers.append(nn.Conv2d(inChannels, outputLayers, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.MakeModel()

class ReshapeLayer(nn.Module):
        def __init__(self, shape):
            self.shape = shape

        def forward(self, x):
            return x.view(*self.shape)

class RoPose_AdvancedBaseline(RoPose_SimpleBaseline):

    def __init__(self):
        self.batchNormMomentum = 0.1
        self.bootstrapStage: nn.Sequential
        self.refStages: List[nn.Sequential] = []
        self.featurePreparationStage: nn.Sequential

        super().__init__()

    def forward(self, x):
        # x = resNetFeatures
        out = self.bootstrapStage(x)

        prepFeatures = self.featurePreparationStage(x)

        outs = []
        concats = []
        outs.append(out)
        for i in range(0, len(self.refStages)):
            concats.append(torch.cat([prepFeatures, out], dim=1))
            out = self.refStages[i](concats[-1])
            outs.append(out)

        return outs

    def FeaturePreparationStage(self):
        ret = []
        ret.append(nn.ConvTranspose2d(in_channels=2048, out_channels=config.advancedFeatureStackSize,
                                      kernel_size=(3, 3), bias=False))
        ret.append(torch.nn.Upsample(size=(config.outputRes[0],config.outputRes[1]), mode="bicubic"))
        self.featurePreparationStage = nn.Sequential(*list(ret)).to(self.device)
        return self.featurePreparationStage

    def RefinementStage(self):
        ret = RoposeRefinementStage()
        self.refStages.append(ret)
        return ret


    def DeConvLayer(self, inChannels: int=256, filterSize: int=256, kernelSize: int=4, stride: int=2, padding: int=1,
                    outputPadding: int=0) -> nn.Sequential:
        ret = []
        ret.append(nn.ConvTranspose2d(in_channels=inChannels, out_channels=filterSize,
                           kernel_size=(kernelSize, kernelSize),
                           stride=(stride, stride), padding=(padding, padding),
                           output_padding=(outputPadding, outputPadding), bias=False
                           ))

        ret.append(nn.BatchNorm2d(filterSize, momentum=self.batchNormMomentum))
        ret.append(nn.ReLU(True))
        return nn.Sequential(*list(ret)).to(self.device)

    def OutputLayer(self, inChannels:int, outputLayers:int):
        ret = nn.Conv2d(inChannels, outputLayers, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        return ret

    def UpsamplingLayer(self, factor: float = config.downsampleFactor):
        ret = torch.nn.Upsample(scale_factor=factor, mode="bicubic")
        return ret

    def BootstrapStage(self):
        bootstrapStage = []
        bootstrapStage.append(self.DeConvLayer(inChannels=2048))
        bootstrapStage.append(self.DeConvLayer())
        bootstrapStage.append(self.DeConvLayer())
        bootstrapStage.append(self.OutputLayer(inChannels=256, outputLayers=config.ropose_detectionLayers))
        ret = nn.Sequential(*list(bootstrapStage)).to(self.device)
        self.bootstrapStage = ret
        return ret

    def DefineModel(self):

        netModel = []

        netModel.append(self.BootstrapStage())
        netModel.append(self.FeaturePreparationStage())

        for i in range(0, config.refineStageCount):
            netModel.append(self.RefinementStage())

        if config.includeUpsampling:
            netModel.append(self.UpsamplingLayer())

        self.netModel = nn.Sequential(*list(netModel)).to(self.device)

class Human_SimpleBaseline(RoPose_SimpleBaseline):
    def __init__(self):
        self.batchNormMomentum = 0.1
        super().__init__()

    def DefineModel(self):
        self.DeConvLayer(inChannels=2048)
        self.DeConvLayer()
        self.DeConvLayer()
        self.OutputLayer(inChannels=256, outputLayers=config.human_detectionLayers)

        if config.includeUpsampling:
            self.UpsamplingLayer()

        self.netModel = nn.Sequential(*list(self.layers)).to(self.device)


class RoPoseFeatureExtractor(PartModel):

    def DefineModel(self):
        pass

