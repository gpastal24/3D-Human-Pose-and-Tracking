from DARLENE_trackRPN.siamRPNBIG import TrackerSiamRPNBIG
from glob import glob
import time

import cv2
import numpy as np

from ultralytics import YOLO

tracker = TrackerSiamRPNBIG('./SiamRPNOTB.model')

imgs_all = sorted(glob('/home/iason/workspace/stationary/saves/imgs2_nuno/*.jpg'))
init = True
model = YOLO('yolov8n.pt') 
t_all = 0
counter = 0
for ii, img_path in enumerate(imgs_all):
    
    if ii > 1000:
        counter += 1
        img = cv2.imread(img_path)
        if init:
            results = model(img, conf=0.3, classes=0, max_det=2)
            for result in results:
                boxes = result.boxes.data.cpu().numpy()
            t1 = time.time()
            bbox = boxes[0][:4]
            bbox = np.array([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]).astype(int)
            tracker.init(img, bbox)
            init = False
        else:
            t1 = time.time()
            bbox = tracker.update(img).astype(int)
        t_all += time.time() - t1
        img = cv2.putText(img, str(counter/t_all), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(255,0,0), thickness=1)
        cv2.imshow('results', img)
        cv2.waitKey(1)

print('FPS: ', len(imgs_all)/t_all)
cv2.destroyAllWindows