from DARLENE_trackRPN.siamRPNBIG import TrackerSiamRPNBIG
from glob import glob
import time

import cv2
import numpy as np


tracker = TrackerSiamRPNBIG('./SiamRPNOTB.model')

# imgs_all = sorted(glob('/home/iason/workspace/stationary/saves/imgs2_nuno/*.jpg'))

vid = cv2.VideoCapture('/dev/video0')
init = True

t_all = 0
counter = 0

while True:
    ret, img = vid.read()
    if not ret:
        break
    if init:
        bbox = cv2.selectROI('select', img)
        cv2.destroyWindow('select')
        bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]]).astype(int)
        tracker.init(img, bbox)
        init = False
    else:
        t1 = time.time()
        bbox = tracker.update(img).astype(int)
    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(255,0,0), thickness=1)
    cv2.imshow('results', img)
    cv2.waitKey(25)
cv2.destroyAllWindows()
vid.release()
