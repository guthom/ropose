# RoPose
This repository contains the core of the RoPose Pose estimation System for industrial Robot arms the designed  CNNs and training/run/validation code/scripts

### Prerequisites 

* Python 3.6 (for Typing etc.)
* Numpy
* OpenCV
* Cuda 10.0 and CuDnn
* PyTorch or/and Keras
* Tensorflow
* termcolor
* Transforms3d
* pytorch-summary
* Pycocotools
* guthoms_helpers https://github.com/guthom/guthoms_helpers (submodule)
* ropose_dataset_tools https://github.com/guthom/ropose_dataset_tools (submodule)
* ropose_greenscreener https://github.com/guthom/ropose_greenscreener (submodule)
* human-pose-estimation https://github.com/microsoft/human-pose-estimation.pytorch (submodule)
* Ultralytics YoloV3 https://github.com/ultralytics/yolov3 (submodule)

## Installing
* Clone the repository
* run git submodule update --init --recursive to get the needed submodules
* Install required packages: pip3 install -r requirements.txt


## Open Source Acknowledgments
This work uses parts from:
* **PyTorch** https://pytorch.org/
* **NumPy** https://www.numpy.org/
* **OpenCV** https://opencv.org/
* **Keras** https://keras.io/
* **TensorFlow** https://www.tensorflow.org/
* **termcolor** https://pypi.org/project/termcolor/
* **Transforms3d** https://matthew-brett.github.io/transforms3d/
* **pytorch-summary** https://github.com/sksq96/pytorch-summary
* **Pycocotools** http://cocodataset.org
* **Ultralytics YoloV3** https://github.com/ultralytics/yolov3
* **Simple Baselines for Human Pose Estimation and Tracking**  https://github.com/microsoft/human-pose-estimation.pytorch
* **

**Thanks to ALL the people who contributed to the projects!**

## Authors

* **Thomas Gulde**

Cognitive Systems Research Group, Reutlingen-University:
https://cogsys.reutlingen-university.de/

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Citation
Please cite the following papers if this code is helpful for your research. 

```bib
@inproceedings{gulde2019roposeReal,
  title={RoPose-Real: Real World Dataset Acquisition for Data-Driven Industrial Robot Arm Pose Estimation},
  author={Gulde, Thomas and Ludl, Dennis and Andrejtschik, Johann and Thalji, Salma and Curio, Crist{\'o}bal},
  booktitle={2019 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2019},
  organization={IEEE}
}

@inproceedings{gulde2018ropose,
  title={RoPose: CNN-based 2D Pose Estimation of industrial Robots},
  author={Gulde, Thomas and Ludl, Dennis and Curio, Crist{\'o}bal},
  booktitle={2018 IEEE 14th International Conference on Automation Science and Engineering (CASE)},
  pages={463--470},
  year={2018},
  organization={IEEE}
}
```

