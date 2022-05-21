from unittest import TestCase
import os
from ropose.net.pytorch.Util import Util
from ropose_dataset_tools.DataSetLoader import LoadDataSet
from guthoms_helpers.filesystem.FileHelper import FileHelper

from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset
from ropose_dataset_tools.DataClasses.Dataset.BoundingBox import BoundingBox
from guthoms_helpers.base_types.Pose2D import Pose2D
import copy

import cv2


class PostprocessingTests(TestCase):

    def setUp(self):
        pass
