import sys, os
import ropose.scripts.BootstrapDL

from ropose.net.pytorch.RoposeNet_SimpleBaseline import RoposNet as RoposNet
from ropose.net.pytorch.Util import Util as util
import keras_config as config
import ropose_dataset_tools.DataSetLoader as loader

datasets = loader.LoadDataSets(config.realDataPath, config.simDataPath, config.mixRealWithSimulation)
datasets = util.SwapDatasetData(datasets)
#datasets = [datasets[0]] * 1000

trainSet, testSet, validationSet = util.SplitDataset(dataset=datasets, checkValidity=True)

net = RoposNet(trainSet=trainSet, testSet=testSet, validationSet=validationSet, netResolution=config.inputRes)
net.learningRate = config.startLearningRate

net.LoadPretrainedModel()

net.TrainModel(trainSet=trainSet[:], testSet=testSet, batch_size=config.batchSize,
               epochs=config.epochs)


