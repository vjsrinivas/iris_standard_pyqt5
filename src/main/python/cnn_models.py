import sys
import os
import yaml
from yaml.loader import Loader
from types import SimpleNamespace
import torch

class ModelOptions:
    CFG_FILE = ''
    WEIGHT_FILE = ''
    IMG_SIZE = (0,0)
    CONF_THRESHOLD = 0.5

    def __init__(self, config_file:str, cfg:str, weight_file:str, img_size:int, ) -> None:
        self.data = yaml.load(open(config_file), Loader=Loader)
        self.CFG_FILE = cfg
        self.WEIGHT_FILE = weight_file
        self.IMG_SIZE = img_size
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            self.half = True
        else:
            self.device = 'cpu'
            self.half = False

    def compile(self):
        _namespace = SimpleNamespace(
            weights = [self.WEIGHT_FILE],
            img_size = self.IMG_SIZE,
            conf_thres = self.CONF_THRESHOLD,
            iou_thres = 0.6,
            device = self.device,
            names = self.data['names'],
            cfg = self.CFG_FILE,
            half = self.half,
            classes = None,
            agnostic_nms = False
        )
        return _namespace

PARENT = './src/main/python/'
YOLOv4_VOC = ModelOptions(
    os.path.join(PARENT, 'cnns/yolo/data/voc.yaml'),
    os.path.join(PARENT, 'cnns/yolo/cfg/yolov4-voc.cfg'),
    os.path.join(PARENT, 'cnns/yolo/runs/exp11/weights/best.pt'),
    416)