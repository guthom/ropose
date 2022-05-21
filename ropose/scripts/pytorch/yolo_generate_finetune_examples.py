from kinematic_tracker.tracking.Tracker import Tracker
from typing import List
from ropose.scripts.pytorch.colropose_base import *
from ropose.net.pytorch.TrainingHyperParams import TrainingHyperParams
from guthoms_helpers.filesystem.FileHelper import FileHelper
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
from guthoms_helpers.common_stuff.ProgressBar import ProgressBar
from ropose_dataset_tools.DataSetLoader import LoadDataSet
from shutil import copyfile

originalModelPath = "/mnt/datastuff/Final_Ropose/best/Yolo.pt"

yolo.LoadPretrainedModel(originalModelPath)
yolo.ActivateFinetuning()

iteration = 0
upsampleOutput = True

outputPath = "/mnt/datastuff/TestExamples/colropose_tracking/"
fineTuneDatasetPath = "/mnt/datastuff/FineTuning/FineTuneExamples/" + str(iteration)
imageDataFilePath = fineTuneDatasetPath + "/finetune.txt"
imageLabelsFilePath = fineTuneDatasetPath + "/labels/"
imageImagesFilePath = fineTuneDatasetPath + "/images/"

DirectoryHelper.CreateIfNotExist(fineTuneDatasetPath)
DirectoryHelper.CreateIfNotExist(imageLabelsFilePath)
DirectoryHelper.CreateIfNotExist(imageImagesFilePath)

roposeDatasets = loader.LoadDataSets(config.roposeEvalDataPath, None, False)

roposeTracker = Tracker(similarityThreshold=0.6, invalidationTimeMs=5000)

trackingRoposeRecoveryCounter = 0

roposeTrainSets: List[YoloDetection] = []

exampleCounter = 0

for datasetIndex in ProgressBar(range(0, roposeDatasets.__len__())):
    dataset = roposeDatasets[datasetIndex]
    if dataset is None:
        # reset tracker for a new dataset
        roposeTracker = Tracker(similarityThreshold=0.6, invalidationTimeMs=5000)
        continue


    rawFrame = cv2.imread(dataset.rgbFrame.filePath)
    rawFrame = cv2.cvtColor(rawFrame, cv2.COLOR_BGR2RGB)

    x, padding, resizeFactor = Util.PrepareRawInput(rawFrame, config.yolo_InputSize, printTestTimers=False)

    roposeDetections, humanDetections, elapsed = DetectYolo(yolo, x)

    yoloFrame = copy.deepcopy(rawFrame)

    bbRopose = []
    if roposeDetections.__len__() > 1:
        test = True

    for detection in roposeDetections:
        bbRopose.append(detection.boundingBox)
        yoloFrame = detection.boundingBox.Draw(yoloFrame, description="robot")

    drawDebugFrame = copy.deepcopy(rawFrame)

    roposeBeforeTrackingRecovery = bbRopose.__len__()
    notRecognizedRopose = roposeTracker.GetNotKnownInstances(bbRopose, predict=True, useHistoryCount=5,
                                                             iouThreshold=0.1)
    for newDet in notRecognizedRopose:
        newDet.Draw(drawDebugFrame, "Robot")
        newYolo = YoloDetection(newDet, config.yolo_RoposeClassNum, 1.0)
        roposeTrainSets.append(newYolo)
        roposeDetections.append(newYolo)

    roPoseRet = Ropose(roPoseNet, rawFrame, roposeDetections, prinTimer=False, upsampleOutput=upsampleOutput)

    # draw ropose
    if roPoseRet is not None:
        supervisions = []
        for keypoints in roPoseRet[0]:
            roposeModel = PoseModelRopose_2D()
            roposeModel.UpdatePoses(keypoints)
            supervisions.append(roposeModel)

        trackingRoposeRecoveryCounter += supervisions.__len__() - roposeBeforeTrackingRecovery
        sims = roposeTracker.AddSupervision(supervisions)
        print("RobotSims: " + str(sims))

    currentFPS = np.round(fpsTracker.FinishRun(), decimals=2)

    print("Tracker FPS: ", str(currentFPS))
    print("--END OF FRAME--")
    print("")

    print("RoPose Recovered:" + str(trackingRoposeRecoveryCounter))

    for i in range(0, len(roposeTrainSets)):
        yoloData = roposeTrainSets[i]
        labelFilePath = imageLabelsFilePath + str(exampleCounter) + ".txt"
        imageFilePath = imageImagesFilePath + str(exampleCounter) + ".png"

        copyfile(dataset.rgbFrame.filePath, imageFilePath)

        with open(imageDataFilePath, 'w+') as file:
            file.write(imageFilePath + '\n')

        with open(labelFilePath, 'w+') as file:
            rawImage = cv2.imread(dataset.rgbFrame.filePath)
            imgWidth = rawImage.shape[1]
            imgHeight = rawImage.shape[0]

            bb = yoloData.boundingBox
            classID = yoloData.predictedClass

            x = bb.midX / imgWidth
            y = bb.midY / imgHeight
            w = bb.size.x / imgWidth
            h = bb.size.y / imgHeight
            string = str(classID) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " "
            file.write(string + '\n')

        exampleCounter += 1

    roposeTrainSets.clear()

print("Finished!")

