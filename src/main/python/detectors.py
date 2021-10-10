import sys
import abc # Abstract Base Classes
import random
import numpy as np

# import from ./src/main/python folder:
sys.path.append('./src/main/python')
import config

class Model(abc.ABC):
    def __init__(self, *network_config) -> None:
        self.__network_config__ = network_config
        self.initialize(*network_config)
    
    @abc.abstractclassmethod
    def run(self, input):
        raise NotImplementedError

    @abc.abstractclassmethod
    def initialize(self, *kwargs):
        raise NotImplementedError

    @abc.abstractclassmethod
    def deinitialize(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def draw(pred):
        raise NotImplementedError

    def __call__(self):
        pred = self.run()
        return pred

# Define functionality for YOLO networks
# Inherit base functions from Model
class YOLOv4(Model):
    def __init__(self, *network_config) -> None:
        self.__device__ = None
        self.__model__ = None
        self.config = network_config[0]
        super().__init__(*network_config)
        self.__last_time__ = 0
        self.__last_label__ = ""

    def initialize(self, *kwargs):
        # assuming kwargs is from the correct given item in config.py:
        from nn.yolo import detect
        from nn.yolo.utils.general import plot_one_box
        self.__run__ = detect.detect
        self.__draw__ = plot_one_box
        self.__model__, self.__device__ = detect.load_model(kwargs[0])
    
    def run(self, input):
        # run should really only return detections, but we can make other helper functions for other outputs from the model
        assert len(input.shape) == 3, "Input given to model is not in proper OpenCV2 image format!"
        pred = self.__run__(self.__model__, input, self.config, self.__device__)
        detections, infer_time, label_string = pred
        self.__last_time__ = infer_time
        self.__last_label__ = label_string
        return detections

    def draw(self, pred, im0):
        # Get names and colors
        random.seed(10)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.config.names))]
        for *xyxy, conf, cls in pred:
            label = '%s %.2f' % (self.config.names[int(cls)], conf)
            self.__draw__(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
        return im0

    def deinitialize(self):
        return super().deinitialize()

    @property
    def inferTime(self):
        return self.__last_time__

    @property
    def detectionLabels(self):
        return self.__last_label__