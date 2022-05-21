from  ropose_dataset_tools.DataClasses.Dataset import Dataset
import tensorflow as tf
from ropose.net.pytorch.Util import Util
import ropose.pytorch_config as config
from pycocotools.coco import COCO

class TensorboardLogger(object):

    def __init__(self, logDir: str):
        """Create a summary writer logging to log_dir."""
        self.logDir = logDir

    def LogEpoch(self, meanLoss, lr, epoch):

        writer = tf.summary.FileWriter(self.logDir)

        summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=meanLoss)])
        writer.add_summary(summary, epoch)
        summary = tf.Summary(value=[tf.Summary.Value(tag="lr", simple_value=lr)])
        writer.add_summary(summary, epoch)
        writer.close()
