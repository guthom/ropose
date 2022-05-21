import os

from ropose.net.pytorch.RoposeNet_SimpleBaseline import RoposNet as RoposNet

import ropose.pytorch_config as config
from ropose.net.pytorch.TrainingHyperParams import TrainingHyperParams

import ropose_dataset_tools.DataSetLoader as loader

roPoseTrain = loader.LoadDataSets(config.realDataPath, config.simDataPath, config.mixRealWithSimulation)
roPoseTest = loader.LoadDataSets(config.roposeEvalDataPath, config.simDataPath,  config.mixRealWithSimulation)
roPoseValidationSet = None

net = RoposNet(trainSet=roPoseTrain,
               testSet=roPoseTest,
               validationSet=roPoseValidationSet,
               netResolution=config.inputRes)

hyperParams = TrainingHyperParams()
hyperParams.modelName = "RoPose"
hyperParams.seed = 1337
hyperParams.epochs = 400
hyperParams.batchSize = 1
hyperParams.optimizer = "SGD"
hyperParams.optimizer_lr = 1e-3
hyperParams.optimizer_momentum = 0.99
#hyperParams.optimizer_amsgrad = True
#hyperParams.scheduler = "multistep"
hyperParams.scheduler = "expo"
hyperParams.scheduler_gamma = 0.001
hyperParams.useBackgroundAugmentation = config.onTheFlyBackgroundAugmentation
hyperParams.useForegroundAugmentation = config.onTheFlyForegroundAugmentation
hyperParams.useRandomErasing = config.useRandomErasing
#hyperParams.scheduler_milestones = [80, 150, 250]
basePath = os.path.join(config.trainedModelBasePath, "SingleTrained")
hyperParams.finalModelDir = os.path.join(basePath, hyperParams.modelName)
hyperParams.PlantSeed()

bestLoss = net.TrainModel(hyperParams, trainSet=roPoseTrain, testSet=roPoseTest)
net.FinishTraining(hyperParams, bestLoss)


