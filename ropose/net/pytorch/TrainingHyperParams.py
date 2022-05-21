import os
from torch import nn, optim, utils

from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, LambdaLR, ReduceLROnPlateau
from guthoms_helpers.pytorch_stuff.lr_scheduler.ExponentialDecay import ExponentialDecay
import ropose.pytorch_config as config
import json
from typing import List, Dict
import datetime
import numpy as np
import random
import torch
from termcolor import colored

class TrainingHyperParams(object):

    def __init__(self):
            self.epochs: int = None
            self.batchSize: int = None
            self.seed: int = None
            self.optimizer: str = None
            self.optimizer_lr: float = config.startLearningRate
            self.optimizer_momentum: float = 0.99
            self.optimizer_amsgrad: bool = False
            self.useBackgroundAugmentation: bool = config.onTheFlyBackgroundAugmentation
            self.useForegroundAugmentation: bool = config.onTheFlyForegroundAugmentation
            self.mixWithSimulationData: bool = config.mixRealWithSimulation
            self.useL2Regularization: bool = False
            self.useRandomErasing: bool = config.useRandomErasing

            self.scheduler: str = None
            self.scheduler_gamma: float = None
            self.scheduler_milestones: List[int] = None

            self.modelName: str = None

            self.finalModelDir: str = None

            self.weightDecay = 1e-5

    def PlantSeed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        # if you are suing GPU
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        os.environ['PYTHONHASHSEED'] = str(self.seed)
        print(colored("Set Random-Seed to: " + str(self.seed), 'blue'))

    def GetOptimizer(self, netModel):
        if self.useL2Regularization:
            weightDecay = self.weightDecay
        else:
            weightDecay = 0.0

        if self.optimizer == "SGD":
            if self.optimizer_momentum > 0.0:
                return optim.SGD(netModel.parameters(), lr=self.optimizer_lr, momentum=self.optimizer_momentum,
                                 weight_decay=weightDecay, nesterov=True)
            else:
                return optim.SGD(netModel.parameters(), lr=self.optimizer_lr, momentum=self.optimizer_momentum,
                                 weight_decay=weightDecay)
        elif self.optimizer == "Adam":
            return optim.Adam(netModel.parameters(), lr=self.optimizer_lr, amsgrad=self.optimizer_amsgrad,
                              weight_decay=weightDecay)
        elif self.optimizer == "AdamW":
            return optim.AdamW(netModel.parameters(), lr=self.optimizer_lr, amsgrad=self.optimizer_amsgrad,
                               weight_decay=weightDecay)
        else:
            raise Exception("Optimizer not supported yet!")

    def GetOptimizerForParameterGroup(self, parameterGroup: List):
        if self.useL2Regularization:
            weightDecay = self.weightDecay
        else:
            weightDecay = 0.0

        if self.optimizer == "SGD":
            if self.optimizer_momentum > 0.0:
                return optim.SGD(parameterGroup, lr=self.optimizer_lr, momentum=self.optimizer_momentum,
                                 weight_decay=weightDecay, nesterov=True)
            else:
                return optim.SGD(parameterGroup, lr=self.optimizer_lr, momentum=self.optimizer_momentum,
                                 weight_decay=weightDecay)
        elif self.optimizer == "Adam":
            return optim.Adam(parameterGroup, lr=self.optimizer_lr, amsgrad=self.optimizer_amsgrad,
                              weight_decay=weightDecay)
        elif self.optimizer == "AdamW":
            return optim.AdamW(parameterGroup, lr=self.optimizer_lr, amsgrad=self.optimizer_amsgrad,
                               weight_decay=weightDecay)
        else:
            raise Exception("Optimizer not supported yet!")


    def GetScheduler(self, optimizer):
        if self.scheduler == "expo":
            return ExponentialDecay(optimizer, gamma=self.scheduler_gamma, milestones=self.scheduler_milestones)
        elif self.scheduler == "multistep":
            return MultiStepLR(optimizer, milestones=self.scheduler_milestones, gamma=self.scheduler_gamma)
        elif self.scheduler == "plateau":
            return ReduceLROnPlateau(optimizer, factor=self.scheduler_gamma, patience=5)
        elif self.scheduler is None:
            return None
        else:
            raise Exception("Scheduler not supported yet!")

    def GetFinalModelName(self, fileEnding: str = ".pt") -> str:
        modelName = self.modelName + "_"
        modelName += "E" + "_" + str(self.epochs) + "_"
        modelName += "BS" + "_" + str(self.batchSize) + "_"

        modelName += "OPTIM" + "_" + self.optimizer + "_"
        modelName += "LR" + "_" + str(self.optimizer_lr) + "_"
        if self.optimizer == "SGD":
            modelName += "MOM" + "_" + str(self.optimizer_momentum) + "_"

        if self.scheduler != None:
            modelName += "SCHED" + "_" + self.scheduler + "_"

        if self.scheduler_gamma != None:
            modelName += "_GAMMA" + "_" + str(self.scheduler_gamma)

        modelName += "_L2Reg" + "_" + str(self.useL2Regularization)

        if self.scheduler == "multistep":
            modelName += "_MILES"
            for milestone in self.scheduler_milestones:
                modelName += "_" + str(milestone)


        modelName += "BA"  + str(self.useBackgroundAugmentation)
        modelName += "_FA" + str(self.useForegroundAugmentation)
        modelName += "_RanEr" + str(self.useRandomErasing)
        modelName += "_MixSim" + str(self.mixWithSimulationData)
        modelName += "_SEED_" + str(self.seed)

        modelName += fileEnding

        return modelName

    def ToJson(self):
        jsonDict = {
            "epochs": self.epochs,
            "batchSize": self.batchSize,
            "seed": self.seed,

            "optimizer": self.optimizer,
            "optimizer_lr": self.optimizer_lr,
            "optimizer_momentum": self.optimizer_momentum,
            "mix_with_simulation_data": self.mixWithSimulationData,
            "use_l2_reg": self.useL2Regularization,
            "use_random_erasing": self.useRandomErasing,

            "background_augmentation": self.useBackgroundAugmentation,
            "foreground_augmentation": self.useForegroundAugmentation,

            "scheduler": self.scheduler,
            "scheduler_gamma": self.scheduler_gamma,
            "scheduler_milestones": self.scheduler_milestones,
            "finalModelName": self.GetFinalModelName()
        }

        return jsonDict

    @classmethod
    def FromDict(cls, dict: Dict):
        ret = cls()
        ret.epochs = dict["epochs"]
        ret.batchSize = dict["batchSize"]
        ret.seed = dict["seed"]
        ret.optimizer = dict["optimizer"]
        ret.optimizer_lr = dict["optimizer_lr"]

        if dict.__contains__("optimizer_momentum"):
            ret.optimizer_momentum = dict["optimizer_momentum"]

        if dict.__contains__("use_l2_reg"):
            ret.useL2Regularization = dict["use_l2_reg"]

        if dict.__contains__("use_random_erasing"):
            ret.useRandomErasing = dict["use_random_erasing"]

        if dict.__contains__("mix_with_simulation_data"):
            ret.mixWithSimulationData = dict["mix_with_simulation_data"]

        ret.scheduler = dict["scheduler"]
        ret.scheduler_gamma = dict["scheduler_gamma"]

        if dict.__contains__("scheduler_milestones"):
            ret.scheduler_milestones = dict["scheduler_milestones"]

        ret.modelName = dict["modelName"]

        ret.finalModelDir = None

    def SaveSummary(self, fileDir: str=None, bestLoss: float = None):

        finalModelName = self.GetFinalModelName(fileEnding=".json")

        if fileDir is None:
            filePath = os.path.join(self.finalModelDir, finalModelName)
        else:
            filePath = os.path.join(fileDir, finalModelName)

        jsonDict = self.ToJson()
        if bestLoss is not None:
            jsonDict["bestLoss"] = bestLoss

        jsonDict["timestamp"] = str(datetime.datetime.now())

        with open(filePath, 'w') as outfile:
            json.dump(jsonDict, outfile, indent=4)


