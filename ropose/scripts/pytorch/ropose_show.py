import os
from ropose.net.pytorch.RoposeNet_SimpleBaseline import RoposNet as RoposNet
from ropose.net.pytorch.Util import Util
import ropose.pytorch_config as config
from ropose_dataset_tools.DataOrganizer import DataOrganizer
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
import matplotlib.pyplot as plt
from ropose.scripts.pytorch.TestMethods import TestRopose

roPoseDatasets = DataOrganizer(dataPath=config.roposeEvalDataPath + "/colropose_eval_006").datasets[0]

net = RoposNet(netResolution=config.inputRes)
net.LoadPretrainedModel(config.roPoseNetWeights)
roPoseValidationSet = None

fps = []

figDir = os.path.join(config.showExamplePath, "ropose")
DirectoryHelper.CreateIfNotExist(figDir)

counter = 0

for dataset in roPoseDatasets:
    rawFrame, keypoints, probs, distance, elapsed = TestRopose(net, dataset, upsampleOutput=False, printTimer=False,
                                                               upsamplingOriginalSize=False)

    fps.append(1 / elapsed)
    print("~FPS: " + str(fps[-1]))

    for gt in dataset.yoloData.keypoints:
        poseImage = Util.DrawPose(rawFrame, gt, config.ropPoseColorsGT)

    for bb in dataset.yoloData.boundingBoxes:
        poseImage = bb.Draw(poseImage)

    poseImage = Util.DrawPose(poseImage, keypoints[0], config.ropPoseColors)

    print("Distance: " + str(distance))
    plt.imshow(poseImage)
    plt.show()

    filename = os.path.join(figDir, "ropose_" + str(counter) + ".jpg")
    counter += 1
    plt.savefig(filename, dpi=300)
    print("Saved figure: " + filename)
    counter += 1







