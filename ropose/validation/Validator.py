import os
import numpy as np
import copy
import math
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import ropose.pytorch_config as config
from inspect import signature
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from scipy.integrate import simpson
import pandas as pd
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D
from ropose_dataset_tools.DataClasses.DetectionTypes.YoloDetection import YoloDetection

from typing import Optional, List, Tuple, Dict
import seaborn as sns

class Validator:

    def __init__(self):
        pass

    @staticmethod
    def CompareYoloDetectionWithGT(detection: YoloDetection, groundTruth: List[YoloDetection],
                                   threshold: float = 0.5) -> Tuple[float, List[YoloDetection]]:
        iou: float = 0.0

        ious: List[float] = []
        for gt in groundTruth:
            ious.append(detection.boundingBox.CalculateIoU(gt.boundingBox))

        if ious.__len__() > 0:
            maxIndex: int = int(np.argmax(ious))

            if ious[maxIndex] > threshold:
                iou = ious[maxIndex]
                del groundTruth[maxIndex]

        for rest in groundTruth:
            ious.append(0.0)

        return iou, groundTruth

    @staticmethod
    def EvaluateFalseNegPos(rawPredictions: List, rawGroundTruth: List, threshold: float = 0.5) \
            -> Tuple[Dict, Dict, Dict]:

        copyPredictions = copy.copy(rawPredictions)
        copyGroundTruth = copy.copy(rawGroundTruth)

        falsePos = {k: 0 for k in range(0, config.yolo_Classes.__len__())}
        falseNeg = {k: 0 for k in range(0, config.yolo_Classes.__len__())}
        corrects = {k: 0 for k in range(0, config.yolo_Classes.__len__())}


        for i in range(0, rawPredictions.__len__()):
            predictions = copyPredictions[i]
            gts = copyGroundTruth[i]

            tempPredictions = copy.copy(predictions)
            for j in range(0, predictions.__len__()):
                prediction = predictions[j]
                #remove prediction cause we used it
                tempPredictions.remove(prediction)
                matched = False
                for gt in gts:
                    matched = gt.Match(prediction, minimumIoU=threshold)
                    if matched:
                        corrects[prediction.predictedClass] += 1
                        #remove the samples from gt
                        gts.remove(gt)
                        break

            copyPredictions[i] = tempPredictions
            copyGroundTruth[i] = gts

        # count leftovers
        for i in range(0, copyPredictions.__len__()):
            predictions = copyPredictions[i]
            gts = copyGroundTruth[i]

            # count false positives
            for pred in predictions:
                falsePos[pred.predictedClass] += 1

            # count false negatives
            for pred in gts:
                falseNeg[pred.predictedClass] += 1

        return corrects, falsePos, falseNeg

    @staticmethod
    def NewEvaluateBinary(rawPredictions: List, rawGroundTruth: List, threshold: float = 0.5) \
            -> Tuple[Dict, Dict]:

        binaryPredictions = {k: [] for k in range(0, config.yoloClassMap.__len__())}
        binaryLabels = {k: [] for k in range(0, config.yoloClassMap.__len__())}

        for i in range(0, rawPredictions.__len__()):
            predictions = copy.deepcopy(rawPredictions[i])
            gts = copy.deepcopy(rawGroundTruth[i])

            predsToRemove = []

            for j in range(0, predictions.__len__()):
                prediction = predictions[j]

                highestMatch = 0
                highestMatchGT = None

                for gt in gts:
                    matched = gt.Match(prediction, minimumIoU=threshold)
                    if matched:
                        if highestMatch < matched:
                            highestMatch = matched
                            highestMatchGT = gt

                if highestMatchGT is not None:
                   binaryPredictions[prediction.predictedClass].append(True)
                   binaryLabels[prediction.predictedClass].append(True)
                   gts.remove(highestMatchGT)
                   predsToRemove.append(prediction)

            for pred in predsToRemove:
                predictions.remove(pred)

            for pred in predictions:
                #handle all to much recognized labels
                binaryPredictions[pred.predictedClass].append(True)
                binaryLabels[pred.predictedClass].append(False)

            for gt in gts:
                #handle not recognized gt labels
                binaryPredictions[gt.predictedClass].append(False)
                binaryLabels[gt.predictedClass].append(True)

        #check data
        for i in range(0, binaryLabels.__len__()):
            assert(binaryPredictions[i].__len__() == binaryLabels[i].__len__())

        return binaryPredictions, binaryLabels

    @staticmethod
    def Calculate_mIoUs(ious: dict, classIndexes: List[int] = None) -> float:

        if classIndexes is None:
            classKeys = ious.keys()
        else:
            classKeys = classIndexes

        counter = 0
        summ = 0

        for classNr in classKeys:
            iou = np.mean(ious[classNr])

            if not np.isnan(iou):
                summ += iou
                counter += 1

        return summ / counter

    @staticmethod
    def Calculate_mAP(aps: dict, classIndexes: List[int] = None) -> float:

        if classIndexes is None:
            classKeys = aps.keys()
        else:
            classKeys = classIndexes

        counter = 0
        summ = 0

        for classNr in classKeys:
            ap = aps[classNr]

            if not np.isnan(ap):
                summ += ap
                counter += 1

        return summ/counter


    @staticmethod
    def Evaluate_PrecissionRecall_Binary_ALL(binaryPredictions: dict, binaryLabels: dict, path: str) -> Dict:

        allBinaryPredictions = {}
        allBinaryPredictions["all"] = []

        allBinaryLabels = {}
        allBinaryLabels["all"] = []

        for key in binaryPredictions:
            allBinaryPredictions["all"].extend(binaryPredictions[key])
            allBinaryLabels["all"].extend(binaryLabels[key])

        return Validator.Evaluate_PrecissionRecall_Binary(allBinaryPredictions, allBinaryLabels, path, ["all"])

    @staticmethod
    def CalculateBinaryPrecisionRecall(predictions: List[bool], labels: List[bool]):

        assert(predictions.__len__() == labels.__len__())

        TP = 0
        FP = 0
        FN = 0

        for i in range(0, predictions.__len__()):

            if predictions[i] == labels[i]:
                TP += 1
            elif predictions[i] is False and labels[i] is True:
                FN += 1
            elif predictions[i] is True and labels[i] is False:
                FP += 1

        prec = TP / (TP + FP)
        rec = TP / (TP + FN)

        if (prec + rec) != 0.0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0

        return prec, rec, f1

    @staticmethod
    def SetLines(lines, colorMap=None):
        for i in range(0, lines.__len__()):
            plt.setp(lines[i], linewidth=2)
            lines[i][0].set_solid_capstyle('round')
            if colorMap is not None:
                lines[i][0].set_color(colorMap[i])

    @staticmethod
    def DefineStandardPlot(xLabel: str, yLabel: str):
        fig, ax = plt.subplots()
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.set_ylabel(yLabel)
        ax.set_xlabel(xLabel)
        # Major ticks every 20, minor ticks every 5
        plt.grid(True)
        major_ticksY = np.arange(0, 1.1, 0.5)
        minor_ticksY = np.arange(0, 1.1, 0.1)
        major_ticksX = np.arange(0, 1.1, 0.1)

        # ax.set_xticks(major_ticksX)
        # ax.set_yticks(major_ticksY)
        # ax.set_yticks(minor_ticksY, minor=True)

        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)

        return fig, ax

    @staticmethod
    def DefineStandardBarPlot(xLabel: str, yLabel: str, min: float =0.0, max: float=1.0):
        fig, ax = plt.subplots()
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.set_ylabel(yLabel)
        ax.set_xlabel(xLabel)
        # Major ticks every 20, minor ticks every 5
        plt.grid(True)
        major_ticksX = np.arange(min, max + 0.1, 0.1)

        # ax.set_xticks(major_ticksX)
        # ax.set_yticks(major_ticksY)
        # ax.set_yticks(minor_ticksY, minor=True)

        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)

        return fig, ax

    @staticmethod
    def SetBars(bars, colorMap=None):
        for i in range(0, bars.__len__()):
            plt.setp(bars[i], linewidth=2)
            bars[i][0].set_solid_capstyle('round')
            if colorMap is not None:
                bars[i][0].set_color(colorMap[i])

    @staticmethod
    def PlotBars(values: List[float], xLabel: str, yLabel: str, labels: Optional[List[str]]=None,
                 min: float =0.0, max: float=1.0, colors: Optional[List[str]]=None):

        sns.set(color_codes=True)
        #Validator.DefineStandardBarPlot(xLabel, ylabel, min, max)
        #data = sns.load_dataset(values)
        data = pd.DataFrame(values)
        ax = sns.barplot(y=labels, x=values, palette=colors)
        for i, v in enumerate(values):
            ax.text(v + 0.05, i + .25, "{:.3f}".format(v), color='black')

        ax.set(xlim=(min, max), ylabel=yLabel, xlabel=xLabel)
        return plt

    @staticmethod
    def PlotF1Scores(f1Scores: List[float], th: List[float], path:str):
        fig, ax = Validator.DefineStandardPlot(xLabel="Threshold Factor", yLabel="F1 Score")

        lines = []
        lines.append(ax.plot(th, f1Scores, label="F1-Scores"))
        Validator.SetLines(lines)

        plt.axvline(x=0.2, color="red", label="F1@0.2")
        plt.axvline(x=0.5, color="darkgreen", label="F1@0.5")
        plt.axvline(x=0.8, color="black", label="F1@0.8")

        plt.savefig(path, dpi=300)
        plt.clf()

    @staticmethod
    def Evaluate_PrecissionRecall_Binary(results: Dict, path: str,
                                         classIndexes: List[int] = None) -> Tuple:

        #Results -> binaryPredictions, binaryLabels

        aps = {k: 0 for k in range(0, config.yoloClassMap.__len__())}

        if classIndexes is None:
            #extract class keys
            classKeys = results[0.0][0].keys()
        else:
            classKeys = classIndexes

        perClassAnalysis = dict()
        precisions = []
        recalls = []
        areas = {}
        f1Scores = []

        ths = []

        #init dicts
        aps = dict()
        perClassResults = dict()
        for classNr in classKeys:
            aps[classNr] = dict()
            perClassResults[classNr] = dict()
            for th in results:
                aps[classNr][th] = dict()
                perClassResults[classNr][th] = dict()
                ths.append(th)


        for classNr in classKeys:
            for th in results:
                binaryPredictions = results[th][0][classNr]
                binaryLabels = results[th][1][classNr]
                #precision, recall = precision_recall_curve(y_true=binaryLabels, binaryPredictions)
                precision, recall, f1 = Validator.CalculateBinaryPrecisionRecall(predictions=binaryPredictions,
                                                                             labels=binaryLabels)

                if th == 0.2:
                    print("F1@0.2: " + str(f1))
                if th == 0.5:
                    print("F1@0.5: " + str(f1))
                if th == 0.8:
                    print("F1@0.8: " + str(f1))

                f1Scores.append(f1)
                perClassResults[classNr][th] = [precision, recall]
                aps[classNr][th] = average_precision_score(binaryLabels, binaryPredictions)

        for classNr in classKeys:

            precisions = []
            recalls = []

            for th in perClassResults[classNr].keys():
                precisions.append(perClassResults[classNr][th][0])
                recalls.append(perClassResults[classNr][th][1])

            filename = os.path.join(path, "pr_curve_class_" + str(classNr) + ".png")

            fig, ax = Validator.DefineStandardPlot(xLabel="Recall", yLabel="Precision")

            lines = []
            recalls = list(reversed(recalls))
            lines.append(ax.step(recalls, precisions, label="P/R class " + config.yolo_Classes[classNr]["name"]))

            #calculate area under curve (AUC)
            area = auc(recalls, precisions)
            print("AUC: " + str(area))
            areas[classNr] = area

            plt.fill_between(recalls, precisions, alpha=0.3)
            Validator.SetLines(lines)

            plt.savefig(filename, dpi=300)
            plt.clf()

            filename = os.path.join(path, "f1Scores_class_" + str(classNr) + ".png")

            Validator.PlotF1Scores(f1Scores=f1Scores, th=ths, path=filename)

        return aps, f1Scores, recalls, precisions, areas

    @staticmethod
    def Evaluate_mAP(precission, recall, step) -> float:
        #calculates area under the precission recall curve
        area = 0

        for i in range(0, precission.__len__()):
            area += precission[i] * recall[i] * step

        return area

    @staticmethod
    def EvaluateAP(rawPredictions: List, rawGroundTruth: List, step: float= 0.01):

        thresholds = np.arange(0, 1.0, step)

        aps = {k: {"P": [], "R": [], "steps": [], "AP": 0} for k in range(0, config.yolo_Classes.__len__())}


        for i in thresholds:
            corrects, falsePos, falseNeg = Validator.EvaluateFalseNegPos(rawPredictions, rawGroundTruth, i)

            precissions = Validator.EvaluatePrecission(corrects, falsePos)
            recalls = Validator.EvaluateRecall(corrects, falseNeg)

            for key in precissions:
                aps[key]["P"].append(precissions[key])
                aps[key]["R"].append(recalls[key])
                aps[key]["steps"].append(i)

        #add mean AP as well
        for key in aps:
            aps[key]["AP"] = Validator.Evaluate_mAP(aps[key]["P"], aps[key]["R"], step)

        return aps

    @staticmethod
    def Plot_mAP(aps: Dict, path: str, classIndexes: List[int] = None) -> List[float]:
        mAPs = []

        DirectoryHelper.CreateIfNotExist(path)

        if classIndexes is None:
            classKeys = aps.keys()
        else:
            classKeys = classIndexes

        for classNr in classKeys:

            filename = os.path.join(path, "pr_curve_class_" + str(classNr) + ".jpg")

            recall = aps[classNr]["R"]
            precision = aps[classNr]["P"]
            step = aps[classNr]["steps"]

            # from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#
            # sphx-glr-auto-examples-model-selection-plot-precision-recall-py

            '''
            # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
            step_kwargs = ({'step': 'post'}
                           if 'step' in signature(plt.fill_between).parameters
                           else {})

            plt.step(recall, precision, color='b', alpha=0.2,
                     where='post')

            plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
            '''

            plt.plot(precision, recall)

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Class ' + str(classNr) + 'Precision-Recall curve: AP={0:0.2f}'.format(
                aps[classNr]["mAP"]))

            plt.savefig(filename)
            plt.clf()

        return mAPs

    @staticmethod
    def EvaluatePrecission(tp, fp) -> Dict:
        #Precision (P) is defined as the number of true positives (tp) over the number of true positives plus
        # the number of false positives (fp).
        precissions = {k: 0 for k in range(0, config.yolo_Classes.__len__())}

        for key in tp:
            if(tp[key]) == 0:
                precissions[key] = 0
            else:
                precissions[key] = tp[key] / (tp[key] + fp[key])

        return precissions

    @staticmethod
    def EvaluateRecall(tp, fn) -> Dict:
        # Recall (R) is defined as the number of true positives (tp) over the number of true positives plus
        # the number of false negatives (fn).
        recalls = {k: 0 for k in range(0, config.yolo_Classes.__len__())}

        for key in tp:
            if(tp[key]) == 0:
                recalls[key] = 0
            else:
                recalls[key] = tp[key] / (tp[key] + fn[key])

        return recalls

    @staticmethod
    def EvaluateSpecifity(fp, tn) -> Dict:
        # Specifity (S) is defined as the number of false Positives (fp) over the number of false Positives plus
        # the number of true negatives
        # see https://www.datascienceblog.net/post/machine-learning/specificity-vs-precision/
        specifities = {k: 0 for k in range(0, config.yolo_Classes.__len__())}

        for key in fp:
            if (fp[key]) == 0:
                specifities[key] = 0
            else:
                specifities[key] = fp[key] / (fp[key] + tn[key])

        return specifities

    @staticmethod
    def EvaluateIoU(pred: List, gt: List):
        # calculates Intersection over union
        ious = {k: [] for k in range(0, config.yoloClassMap.__len__())}

        evalGT = copy.copy(gt)

        for i in range(0, pred.__len__()):
            prediction = pred[i]
            groundTruth = evalGT[i]

            for j in range(0, prediction.__len__()):
                if j < groundTruth.__len__():
                    singlePred = prediction[j]
                    singleGT = groundTruth[j]
                    iou = singlePred.boundingBox.CalculateIoU(singleGT.boundingBox)
                    ious[singleGT.predictedClass].append(iou)

        return ious


    @staticmethod
    def EvaluatePCP():
        raise Exception("Not Implemented!")

    @staticmethod
    def EvaluatePCK(pred: np.array, gt: np.array, refIndexes: Tuple[int, int], setPixTH=None,
                     pckTH: float = 0.5) -> np.array:
        #evals Probability of Correct Keypoints - PCK

        dist = Validator.EvaluateMPJPE(pred, gt)

        numKeypoints = gt.shape[1]
        numExamples = gt.shape[0]

        if setPixTH is None:
            refTHs = np.linalg.norm(gt[:, refIndexes[0]] - gt[:, refIndexes[1]], axis=1) * pckTH
            refTHs = np.repeat(refTHs, numKeypoints)
        else:
            refTHs = np.repeat(setPixTH, numKeypoints)
            refTHs = np.stack([[refTHs] * numExamples])

        refTHs = np.reshape(refTHs, dist.shape)
        count = np.less(dist, refTHs)

        count = np.count_nonzero(count, axis=0)

        pcks = count/numExamples

        return pcks

    @staticmethod
    def EvaluatePCKbb(pred: np.array, gt: np.array, pckTH: float = 0.5) -> np.array:
        # evals Probability of Correct Keypoints - PCK

        dist = Validator.EvaluateMPJPE(pred, gt)

        numKeypoints = gt.shape[1]
        numExamples = gt.shape[0]

        #calculate refTh based on bounding Box
        refTHs = []
        for i in range(0, gt.shape[0]):
            bb = BoundingBox2D.CreateBoundingBox(gt[i, :, :])
            refTHs.append(np.array([bb.DiagLength() * pckTH]))

        refTHs = np.repeat(refTHs, numKeypoints, axis=1)
        count = np.less(dist, refTHs)

        count = np.count_nonzero(count, axis=0)

        pcks = count / numExamples

        return pcks

    @staticmethod
    def EvaluatePCKStatic(pred: np.array, gt: np.array, distanceTH) -> np.array:
        # evals Probability of Correct Keypoints - PCK

        dist = Validator.EvaluateMPJPE(pred, gt)

        numKeypoints = gt.shape[1]
        numExamples = gt.shape[0]

        #calculate refTh based on base segment length
        refTHs = []
        for i in range(0, gt.shape[0]):
            refTHs.append(np.array([distanceTH]))

        refTHs = np.repeat(refTHs, numKeypoints, axis=1)
        count = np.less(dist, refTHs)

        count = np.count_nonzero(count, axis=0)

        pcks = count / numExamples

        return pcks

    @staticmethod
    def EvaluatePCKs(pred: np.array, gt: np.array, refIndexes, setPixTH=None, maxDist=0.5, step=0.025):
        pckTH = np.arange(0, maxDist, step)
        pckTH = np.round(pckTH, decimals=4)
        pcks = dict()

        if setPixTH is not None:
            pckh = Validator.EvaluatePCK(pred, gt, refIndexes=refIndexes, setPixTH=setPixTH)
            pcks[setPixTH] = pckh
            return pcks
        else:
            for TH in pckTH:
                pckh = Validator.EvaluatePCK(pred, gt, refIndexes=refIndexes, setPixTH=setPixTH, pckTH=TH)
                pcks[TH] = pckh

        return pcks

    @staticmethod
    def EvaluatePCKsBB(pred: np.array, gt: np.array, maxDist=0.5, step=0.025):
        pckTH = np.arange(0, maxDist, step)
        pckTH = np.round(pckTH, decimals=4)
        pcks = dict()

        for TH in pckTH:
            pckh = Validator.EvaluatePCKbb(pred, gt, pckTH=TH)
            pcks[TH] = pckh

        return pcks

    @staticmethod
    def EvaluatePCKsStatic(pred: np.array, gt: np.array, maxDist=10, step=0.025):
        pckTH = np.arange(0, 1, step)
        pckTH = np.round(pckTH, decimals=4)
        pcks = dict()

        for TH in pckTH:
            pckh = Validator.EvaluatePCKStatic(pred, gt, distanceTH=maxDist*TH)
            pcks[TH] = pckh

        return pcks

    @staticmethod
    def EvaluateMPJPE(pred: np.array, gt: np.array):
        #evals Mean Per Joint Position Error - MPJPE
        #does not take care of False pos/negatives!
        dist = np.linalg.norm(gt-pred, axis=2)

        return dist


    @staticmethod
    def PlotViolinEval(distances: np.array, maxYVal: Optional[float]=None, YStepSize:Optional[float]=None,
                       labels: Optional[List[str]]=None):

        #create labels if not there
        if labels is None:
            labels = [k for k in range(0, distances.shape[1])]

        errors = distances

        decimals = 3

        means = np.round(np.mean(distances, axis=0), decimals)
        print("Mean = " + str(means))

        maxs = np.round(np.round(np.max(distances, axis=0), decimals))
        print("Max = " + str(maxs))

        mins = np.round(np.round(np.min(distances, axis=0), decimals))
        print("Min = " + str(mins))


        meds = np.round(np.median(distances, axis=0), decimals)
        print("Median = " + str(meds))

        vars = np.round(np.var(distances, axis=0), decimals)
        print("Variance = " + str(vars))

        dev = np.round(np.std(distances, axis=0), decimals)
        print("StadardDeviation = " + str(dev))

        '''
        print("\\begin{tabular}{|r|c|c|c|c|c|c|c|}"
              "\cline{2-8}"
              "\multicolumn{1}{r|}{} & J0	&  J1 	&  J2 	&  J3 	& 	J4 	&	J5 	&	J6 \\\\ \hline"
              "\multicolumn{0}{|r|}{max [pix]}	 	&" + str(maxs[0]) + "&" + str(maxs[1]) + "&" + str(maxs[2]) + "&" +  str(maxs[3]) + "&" +  str(maxs[4]) + "&" +  str(maxs[5]) + "&" +  str(maxs[6]) + "\\\\ \hline"
              "\multicolumn{0}{|r|}{min [pix]}	 	&" + str(mins[0]) + "&" + str(mins[1]) + "&" + str(mins[2]) + "&" +  str(mins[3]) + "&" +  str(mins[4]) + "&" +  str(mins[5]) + "&" +  str(mins[6]) + "\\\\ \hline"
             "\multicolumn{0}{|r|}{mean [pix]}	 	&" + str(means[0]) + "&" + str(means[1]) + "&" + str(means[2]) + "&" +  str(means[3]) + "&" +  str(means[4]) + "&" +  str(means[5]) + "&" +  str(means[6]) + "\\\\ \hline"
             "\multicolumn{0}{|r|}{median [pix]}	 	&" + str(meds[0]) + "&" + str(meds[1]) + "&" + str(meds[2]) + "&" +  str(meds[3]) + "&" +  str(meds[4]) + "&" +  str(meds[5]) + "&" +  str(meds[6]) + "\\\\ \hline"
             "\multicolumn{0}{|r|}{$\sigma$ [pix]}	 	&" + str(dev[0]) + "&" + str(dev[1]) + "&" + str(dev[2]) + "&" +  str(dev[3]) + "&" +  str(dev[4]) + "&" +  str(dev[5]) + "&" +  str(dev[6]) + "\\\\ \hline"
             "\end{tabular}"
              )
        '''
        # plot violin plot

        labelFontSize = None
        tickFintSize = None

        if YStepSize is None:
            stepSize = np.round(maxYVal/10, decimals=0)
        else:
            stepSize = YStepSize

        violinWidth = 0.95

        fig, ax = plt.subplots()

        violin_parts = ax.violinplot(errors, showmeans=True, showmedians=True, widths=violinWidth, points=2000)
        sampleSize = errors[0].__len__()
        #ax.set_title('Jointwise Pixel-error N=' + str(sampleSize))

        ax.yaxis.grid(True)
        ax.set_xticks([x + 1 for x in range(0, sampleSize)])

        if maxYVal is not None:
            yTicks = [0]
            yTicks.extend([y for y in np.arange(0, maxYVal+stepSize, stepSize)])
            ax.set_yticks(yTicks)

            ax.set(ylim=(-1, maxYVal))

        #ax.set_xlabel('Joints')
        ax.set_ylabel('Absolute Error [pix]', fontsize=labelFontSize)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(tickFintSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(tickFintSize)

        plt.setp(ax, xticks=[y + 1 for y in range(0, sampleSize)],
                 xticklabels=labels)

        plt.setp(violin_parts['cmeans'], edgecolor='red')
        plt.setp(violin_parts['cmedians'], edgecolor='black')

        return plt


    @staticmethod
    def PlotmAPs(mAPs: Dict, classIndexes: List[int] = None):

        plt.style.use('fivethirtyeight')

        if classIndexes is None:
            classKeys = mAPs.keys()
        else:
            classKeys = classIndexes

        fig, ax = plt.subplots(figsize=(10, 20))

        counter = 0

        x = []
        #init x
        for th in mAPs[list(mAPs.keys())[0]].keys():
            x.append(th)

        #inti ys
        y = list()
        for i in range(0, len(config.yolo_Classes)):
            y.append(list())

        for key in mAPs.keys():
            #extract values
            for th in mAPs[key].keys():
                y[key].append(mAPs[key][th])

        fig, ax = Validator.DefineStandardPlot(xLabel="Threshold Factor", yLabel="mAP")

        lines = []
        if classIndexes is None:
            for line in y:
                lines.append(ax.plot(x, line))
        else:
            for i in classIndexes:
                line = y[i]
                lines.append(ax.plot(x, line))

        Validator.SetLines(lines)

        return plt


    @staticmethod
    def PlotPCKs(pckh: Dict, index: int=None, all: bool = False, title: str ="PCK", xStep:float = 0.05, yStep=0.1,
                 labels: Optional[List[str]]=None, showLegend:bool=True, legendLOC:str='lower right',
                 inklCombined: bool = False):

        fig, ax = Validator.DefineStandardPlot(xLabel="Threshold Factor", yLabel="PCK")

        values = []
        if index is None:
            for i in range(0, pckh[0.0].__len__()):
                values.append([])
        else:
            values.append([])

        steps = []

        if all:
            values = [[]]
            for entry in pckh:
                values[-1].append(np.mean(pckh[entry]))
                steps.append(entry)

        else:
            for entry in pckh:
                if index is None:
                    for i in range(0, pckh[entry].__len__()):
                        values[i].append(pckh[entry][i])
                else:
                    values[-1].extend([pckh[entry][index]])

                steps.append(entry)

        lines = []

        if not inklCombined or all:
            for i in range(0, values.__len__()):
                if labels is not None:
                    lines.append(ax.plot(steps, values[i], label=labels[i]))
                else:
                    lines.append(ax.plot(steps, values[i]))
        else:
            maxes = []
            mins = []
            combinedVals = []
            for entry in pckh:
                combinedVals.append(np.mean(pckh[entry]))
                maxes.append(np.max(pckh[entry]))
                mins.append(np.min(pckh[entry]))

            lines.append(ax.plot(steps, combinedVals, label="combined", color="blue"))
            ax.fill_between(steps, maxes, mins, alpha=0.1)

            for i in range(0, values.__len__()):

                printer = "["
                for j in range(0, len(values[i])):
                    printer += str(values[i][j]) + ", "

                printer += "]"

                print(printer)

                if labels is not None:
                    lines.append(ax.plot(steps, values[i], label=labels[i], alpha=0.4))
                else:
                    lines.append(ax.plot(steps, values[i]))

        plt.axvline(x=0.2, color="red")
        plt.axvline(x=0.5, color="darkgreen")
        plt.axvline(x=0.8, color="black")

        Validator.SetLines(lines)

        #legend
        if showLegend:
            plt.legend(loc=legendLOC)

        return plt

    @staticmethod
    def PlotPCKsMean(pckh: Dict, title: str ="PCK", xStep:float = 0.05, yStep=0.1, label: str="Combined",
                     showLegend:bool=True, legendLOC:str='lower right'):

        fig, ax = Validator.DefineStandardPlot(xLabel="Threshold Factor", yLabel="PCK")

        values = []
        maxes = []
        mins = []
        steps = []

        for entry in pckh:
            values.append(np.mean(pckh[entry]))
            maxes.append(np.max(pckh[entry]))
            mins.append(np.min(pckh[entry]))

            steps.append(entry)

        plt.axvline(x=0.2, color="red")
        plt.axvline(x=0.5, color="darkgreen")
        plt.axvline(x=0.8, color="black")

        lines = []
        lines.append(ax.plot(steps, values, label=label, color="blue"))
        lines.append(ax.plot(steps, maxes, alpha=0.5, color="blue"))
        lines.append(ax.plot(steps, mins, alpha=0.5, color="blue"))

        ax.fill_between(steps, maxes, mins, alpha=0.2)

        Validator.SetLines(lines)

        #legend
        if showLegend:
            plt.legend(loc=legendLOC)

        return plt

    @staticmethod
    def PlotPCKsSimple(ths: List[float], values: List[np.array], labels: List[str], yLabel: str = "PCK",
                     showLegend: bool = True, legendLOC: str = 'lower right', colors: Optional[List[str]]=None):

        fig, ax = Validator.DefineStandardPlot(xLabel="Threshold Factor", yLabel=yLabel)

        plt.axvline(x=0.2, color="red")
        plt.axvline(x=0.5, color="darkgreen")
        plt.axvline(x=0.8, color="black")

        lines = []
        for i in range(0, values.__len__()):
            lines.append(ax.plot(ths, values[i], label=labels[i]))

        Validator.SetLines(lines, colorMap=colors)

        # legend
        if showLegend:
            plt.legend(loc=legendLOC)

        return plt