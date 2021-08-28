# Standard PyQt5 Application

**Purpose:** This code base is for students looking to jumpstart a standard demo application for IRIS systems. It offers controls on the left-hand side, including ability to specify presets as well as user-defined inputs and outputs. This application is intended for use in primarily imaging situations and uses OpenCV functions and `pixMap` from PyQt5 for displaying video and images. For example, if you needed to showcase a convolutional neural network (CNN), you could use this application as a template and add your own custom functionality.

**Machine Learning Fork:** This fork of the master focuses on implementing convolutional neural networks (CNNs) for object detection. For now, the default detector is a PyTorch implementation of YOLOv4 trained on VOC2007+2012.

<p align="center">
  <img width="460" height="300" src="demo.gif">
</p>

## Requirements

**Python version:** 3.6 (tested) but 3.7 should work

**Python packages:**
```
altgraph==0.17
dataclasses==0.8
fbs==0.9.9
future==0.18.2
macholib==1.15
numpy==1.19.5
opencv-python==4.5.3.56
pefile==2021.5.24
Pillow==8.3.1
PyInstaller==3.4
PyQt5==5.9.2
sip==4.19.8
torch==1.8.0
torchvision==0.9.0
typing-extensions==3.10.0.0

```

## Setup
1. Clone the repo: `git clone https://github.com/vjsrinivas/iris_standard_pyqt5.git`
2. **(Highly Recommended)** Create a virutal Python environment: `python3 -m venv dev`
3. **(If you made a virtual environment)** Activate environment: `source dev/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install PyTorch: `bash torch_install.sh `
    - Note: if you do not have a CUDA-compatiable GPU, remove the `+cu102` from script
5. Download [pretrained model](https://drive.google.com/file/d/1WeRV6fLANM5qJ31aN8RY4ct8d5zJX4XG/view?usp=sharing) folder and place it in the following directory `src/main/python/cnns/yolo/runs/` 
6. Launch application: `fbs run` (User has to be on top-level folder, where the `src` folder is visible) 
