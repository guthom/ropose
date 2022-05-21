from  ropose_dataset_tools.DataClasses.Dataset import Dataset
import tensorflow as tf
from typing import List
from ropose.net.pytorch.Util import Util
from pycocotools.coco import COCO
import ropose.pytorch_config as config
from ropose.net.pytorch.Util import Util

class TensorboardLogger(object):

    def __init__(self, logDir: str, model: 'NetBase', datasets: List['Dataset']):
        """Create a summary writer logging to log_dir."""
        self.logDir = logDir
        self.model = model
        self.annoFile = '{}/annotations/instances_{}.json'.format(config.cocoPath, "train2017")
        self.coco = COCO(self.annoFile)
        self.datasets = []
        for dataset in datasets:
            if dataset.annotations is not None:
                dataset.backgroundMask = self.coco.annToMask(dataset.annotations)
            self.datasets.append(dataset)
        self.datasets = datasets

    def LogEpoch(self, meanLoss, lr, epoch):

        counter = 0
        for dataset in self.datasets:
            imgs, bufs = Util.PlotColPoseSet(dataset=dataset, neuralNet=self.model, upsampling=True,
                                          preprocessFunction=self.model.PreprocessInput)

            # tfImg = tf.image.decode_png(buf.getvalue(), channels=4)
            # tfImg = tf.expand_dims(tfImg, 0)
            for i in range(0, len(imgs)):
                buf = bufs[i]
                img = imgs[i]

                image = tf.Summary.Image(height=1000,
                                         width=1000,
                                         colorspace=3,
                                         encoded_image_string=buf.getvalue())
                buf.close()

                writer = tf.summary.FileWriter(self.logDir)

                summary = tf.Summary(value=[tf.Summary.Value(tag="result_" + str(counter), image=image)])
                writer.add_summary(summary, epoch)
                counter += 1

        counter = 0
        for loss in meanLoss:
            summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss_" + str(counter), simple_value=loss)])
            writer.add_summary(summary, epoch)
            counter += 1

        summary = tf.Summary(value=[tf.Summary.Value(tag="lr", simple_value=lr)])
        writer.add_summary(summary, epoch)
        writer.close()
