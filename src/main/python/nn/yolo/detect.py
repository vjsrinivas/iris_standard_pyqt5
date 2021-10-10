import time
from pathlib import Path

import sys
sys.path.append('./src/main/python/nn/yolo')

import torch
from utils.general import (
    non_max_suppression, scale_coords, xyxy2xywh, plot_one_box)
from utils.torch_utils import select_device, time_synchronized
from models.models import *
from models.experimental import *
from utils.datasets import *
from utils.general import *

import time

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def load_model(opt):
    weights, imgsz, cfg = opt.weights, opt.img_size, opt.cfg

    # Initialize
    device = select_device(opt.device)
    half = opt.half

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    try:
        model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    except:
        load_darknet_weights(model, weights[0])
    
    model.to(device).eval()
    if half:
        model.half()  # to FP16
    
    return model, device

def detect(model, frame, opt, device):
    t1 = time.time()
    imgsz, names, half = opt.img_size, opt.names, opt.half

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    # Padded resize
    img = letterbox(frame, new_shape=imgsz, auto=False, scaleFill=False)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', frame
        s += 'Input size: %gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
    
    t2 = time.time()
    inference_speed = 1/(t2-t1)
    return det, inference_speed, s
    #return im0, inference_speed, s