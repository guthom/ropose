from ropose_dataset_tools.DataClasses.Dataset import Dataset
import tensorflow as tf
from ropose.net.pytorch.Util import Util
import ropose.pytorch_config as config
from pycocotools.coco import COCO

class TensorboardLogger(object):

    def __init__(self, logDir: str, model: 'NetBase', dataset: Dataset=None):
        """Create a summary writer logging to log_dir."""
        self.logDir = logDir
        self.model = model

        self.annoFile = '{}/annotations/instances_{}.json'.format(config.cocoPath, "val2017")
        self.coco = COCO(self.annoFile)

        if dataset is not None:
            if dataset.annotations is not None:
                dataset.backgroundMask = self.coco.annToMask(dataset.annotations)

        self.dataset = dataset

    def LogEpoch(self, meanLoss, lr, epoch, trainLoss=None):

        writer = tf.summary.create_file_writer(self.logDir)
        with writer.as_default():
            tf.summary.scalar("lr", lr, epoch)
            tf.summary.scalar("test_loss", meanLoss, epoch)

            if trainLoss is not None:
                tf.summary.scalar("train_loss", trainLoss, epoch)
        writer.close()

        if self.dataset is not None:

            '''
            img, buf = Util.PlotRoPoseSet(dataset=self.dataset, neuralNet=self.model, upsampling=True,
                                          preprocessFunction=self.model.PreprocessInput)

            # tfImg = tf.image.decode_png(buf.getvalue(), channels=4)
            # tfImg = tf.expand_dims(tfImg, 0)

            image = tf.Summary.Image(height=1000,
                                     width=1000,
                                     colorspace=3,
                                     encoded_image_string=buf.getvalue())
            buf.close()

            summary = tf.Summary(value=[tf.Summary.Value(tag="result", image=image)])
            writer.add_summary(summary, epoch)
            '''
