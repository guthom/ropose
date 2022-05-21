import torch
from torch import nn
import ropose.pytorch_config as config

class SpatialDistributionLoss(nn.Module):
    def __init__(self, maxDistance: float,  weight: float=1):
        super().__init__()
        self.weight = weight
        self.criterion = nn.MSELoss(reduction='mean')
        self.maxDistance = torch.tensor(maxDistance)
        self.th = torch.tensor(config.spatialDistributionLossTH)


    def CalcMeanDistanceOfHeatmap(self, heatmap):

        maxVal = heatmap.max()
        if maxVal == 0:
            return 0

        maxItem = torch.argmax(heatmap)

        y = torch.floor(maxItem / heatmap.shape[0])
        x = maxItem - y*heatmap.shape[0]

        maxItem = torch.tensor([y, x]).to(heatmap.device)

        maxItem = torch.unsqueeze(maxItem, dim=0).type(torch.float)

        # thresholding the heatmap to be not so strict
        compareVal = torch.multiply(maxVal, self.th)
        temp = torch.where(heatmap > compareVal, 1.0, 0.0)
        nonZeros = torch.nonzero(temp).type(torch.float)

        #distance to max item
        #parse to float
        distances = torch.cdist(nonZeros, maxItem)

        if distances.shape[1] == 0:
            return 0

        meanVal = torch.mean(distances)

        if torch.isnan(meanVal):
            return 1.0
        maxDistance = torch.tensor(self.maxDistance).to(heatmap.device)
        meanVal /= maxDistance
        return meanVal


    def forward(self, output, target):
        batches = output.size(0)
        joints = output.size(1)

        loss = 0.0
        for i in range(0, batches):
            for j in range(0, joints):
                predDist = self.CalcMeanDistanceOfHeatmap(output[i, j, :, :])
                gtDist = self.CalcMeanDistanceOfHeatmap(target[i, j, :, :])

                loss += torch.sqrt(torch.pow(gtDist-predDist, torch.tensor(2).to(output.device)))

        loss = (loss / joints / batches) * self.weight
        return loss
