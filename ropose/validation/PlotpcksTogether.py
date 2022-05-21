import os

import matplotlib.pyplot as plt
import numpy as np

from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
from guthoms_helpers.filesystem.FileHelper import FileHelper
from ropose.validation.Validator import Validator

filePath = "/mnt/datastuff/Evaluation/RoPose_Keypoints/new_last/OverallData.pickle"
outputPath = os.path.join(FileHelper.GetFilePath(filePath), "OverallPlots")
DirectoryHelper.CreateIfNotExist(outputPath)
data = FileHelper.DePickleData(filePath)

labels = ["A", "B", "C", "D", "E"]

targetDatas = ["pcks"]

for targetData in targetDatas:
    fig, ax = Validator.DefineStandardPlot(xLabel="Threshold Factor", yLabel="mPCK")
    lines = []
    counter = 0
    outputPlotPath = os.path.join(outputPath, targetData + ".png")
    plt.axvline(x=0.2, color="red")
    plt.axvline(x=0.5, color="black")

    for entry in data:
        print(labels[counter] + ": " + str(entry["settings"]))
        print(np.mean(entry["fps"]))
        plotData = entry[targetData]

        print("PCK@0.2: " + str(np.mean(plotData[0.2])))
        print("PCK@0.5: " + str(np.mean(plotData[0.5])))

        x = []
        y = []
        for th in plotData.keys():
            x.append(th)
            y.append(np.mean(plotData[th]))

        lines.append(ax.plot(x, y, label=labels[counter]))

        counter += 1

    Validator.SetLines(lines)

    plt.legend()
    plt.savefig(outputPlotPath, dpi=300)
    plt.clf()


outputPlotPath = os.path.join(outputPath, "absoluteViolinUpsampling.png")
distances = []
for entry in data:
    flatten = []
    counter = 0
    for a in entry["mpjpe"].flatten():
        flatten.append(a)
        if a >= 1000:
            counter += 1
            #print(counter)


    distances.append(np.array(flatten))


plt = Validator.PlotViolinEval(distances=np.transpose(distances), maxYVal=60.0, labels=labels)
plt.savefig(outputPlotPath, bbox_inches='tight', dpi=300,
            format="pdf")
plt.savefig(outputPlotPath, bbox_inches='tight', dpi=300)
plt.clf()