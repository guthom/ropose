import sys, os
import torch.multiprocessing as mp

from ropose.net.pytorch.RoposeNet_SimpleBaseline import RoposNet as RoposNet
from ropose.net.pytorch.Yolo import Yolo
from ropose.net.pytorch.Util import Util, Timer
import ropose.pytorch_config as config
import copy

from ropose.scripts.pytorch.TestMethods import DetectYolo, DetectHumans, DetectRopose
from guthoms_helpers.input_provider.ImageDirectory import ImageDirectory
from guthoms_helpers.common_stuff.FPSTracker import FPSTracker
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
import cv2

roposeTestTimer = Timer()
humanTestTimer = Timer()

def Ropose(roPoseNet, rawFrame, roposeDetections, mpRet):
    roposeTestTimer.Start()
    keypoints = probs = []
    if roposeDetections.__len__() > 0:
        keypoints, probs, elapsed, pred, paddings, upsampledHeatmaps = DetectRopose(roPoseNet, rawFrame,
                                                                                    roposeDetections,
                                                                                    upsampleOutput=False,
                                                                                    printTimer=True)

    elapsed = roposeTestTimer.Stop()
    #mpRet["ropose"] = (keypoints, probs, elapsed)
    return (keypoints, probs, elapsed)

def HumanPose(humanNet, rawFrame, humanDetections, mpRet):
    humanTestTimer.Start()
    keypoints = probs = []
    if humanDetections.__len__() > 0:
        keypoints, probs, elapsed, pred, paddings, upsampledHeatmaps = DetectHumans(humanNet, rawFrame,
                                                                                     humanDetections,
                                                                                     upsampleOutput=False,
                                                                                     printTimer=True)

    elapsed = humanTestTimer.Stop()
    #mpRet["human"] = (keypoints, probs, elapsed)
    return (keypoints, probs, elapsed)


def PrepareInput(rawFrame):
    rawFrame = cv2.cvtColor(rawFrame, cv2.COLOR_BGR2RGB)
    return rawFrame

if __name__ == "__main__":

    mp.set_start_method('spawn')
    directory = config.roposeEvalDataPath + "colropose_eval_005/depthcam1/rgb0"
    dirInput = ImageDirectory(dirPath=directory, sort=True, preprocessingFunction=PrepareInput)
    resDir = config.outputDir + "/ropose_demo/"
    DirectoryHelper.CreateIfNotExist(resDir)

    yolo = Yolo(netResolution=config.yolo_InputSize)
    yolo.LoadPretrainedModel(config.roPoseYoloWeights)

    roPoseNet = RoposNet(netResolution=config.inputRes)
    roPoseNet.ShareMemory()
    roPoseNet.LoadPretrainedModel(config.roPoseNetWeights)

    fps = []

    fpsTracker = FPSTracker()

    counter = 0
    upsampleOutput = False
    printTestTimers = False

    #multiprocessing stuff
    manager = mp.Manager()
    mpRet = manager.dict()
    mpRet["ropose"] = None
    mpRet["human"] = None

    for i in range(0, 200):
        testTimer = Timer()

        testTimer.Start("Loading")

        rawFrame = dirInput.GetData()

        elapsed = testTimer.Stop(printTestTimers)

        allEllapsed = 0

        allEllapsed += elapsed

        #yolo detection
        testTimer.Start("Yolo")
        roposeDetections, humanDetections, elapsed = DetectYolo(yolo, rawFrame, printTimer=False)
        elapsed = testTimer.Stop(printTestTimers)

        allEllapsed += elapsed
        drawFrame = copy.copy(rawFrame)

        for detection in roposeDetections:
            drawFrame = detection.boundingBox.Draw(drawFrame, color=[255, 0.0, 0.0], description="Robot")

        for detection in humanDetections:
            drawFrame = detection.boundingBox.Draw(drawFrame, color=[0.0, 0.0, 255], description="Human")

        roPoseRet = Ropose(roPoseNet, rawFrame, roposeDetections, mpRet)

        #draw ropose
        if roPoseRet is not None:
            for keypoints in roPoseRet[0]:
                Util.DrawPose(drawFrame, keypoints, config.ropPoseColors)

            allEllapsed += roPoseRet[2]

        fps.append(1 / allEllapsed)
        #print("~FPS: " + str(np.mean(fps)))

        filename = os.path.join(resDir, "ropose_" + str(counter) + ".png")

        drawFrame = cv2.cvtColor(drawFrame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, drawFrame)

        print("Tracker FPS: ", str(fpsTracker.FinishRun()))

        print("Saved figure: " + filename)
        counter += 1







