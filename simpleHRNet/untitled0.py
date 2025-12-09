# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:22:04 2021

@author: giann
"""

import os
import sys
import argparse
import ast
import cv2
import time
import torch
from vidgear.gears import CamGear
import numpy as np
from models.detectors import YOLOv3

sys.path.insert(1, os.getcwd())
from models.poseresnet import PoseResNet
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations

model=PoseResNet(50, 17, 0.1)
yolo_model_def="./models/detectors/yolo/config/yolov3-tiny.cfg"
yolo_class_path="./models/detectors/yolo/data/coco.names"
yolo_weights_path="./models/detectors/yolo/weights/yolov3-tiny.weights"
detector=YOLOv3.YOLOv3(yolo_model_def,yolo_class_path,yolo_weights_path,classes='person')
print(detector)