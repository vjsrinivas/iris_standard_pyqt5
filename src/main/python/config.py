from types import SimpleNamespace
import os
import sys
import torch
import yaml
from yaml.loader import Loader

if torch.cuda.is_available():
    __device__ = 'cuda:0' # force only one (first) gpu!
    __half__ = True
else:
    __device__ = 'cpu'
    __half__ = False

YOLO_ROOT = "./src/main/python/nn/yolo"

# Helper function to read yaml data for things like class names and data paths
def __yolo_yaml_data__(filename):
    return yaml.load(open(filename), Loader=Loader)

YOLOV4_VOC = SimpleNamespace(
    cfg = os.path.join(YOLO_ROOT, 'cfg/yolov4-voc.cfg'),
    weights = [os.path.join(YOLO_ROOT, 'runs/exp11/weights/best.pt')],
    img_size = 416,
    conf_thres = 0.5,
    iou_thres = 0.6,
    device = __device__,
    names = __yolo_yaml_data__(
        os.path.join(YOLO_ROOT,'data/voc.yaml')
    )['names'],
    half = "", #FP16 mode
    classes = None, # filter in certain classes
    agnostic_nms = False

)