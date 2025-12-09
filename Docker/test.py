import cv2
import numpy as np

img = np.ones((640,640,3)).astype(np.uint8)*255
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),15]
result, imgencode = cv2.imencode('.jpg', img, encode_param)

