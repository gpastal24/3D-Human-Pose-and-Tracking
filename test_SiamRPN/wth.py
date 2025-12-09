import numpy as np
def bb_intersection_over_union(boxAA, boxBB):
        """
        Calculate IoU between two boxes
        boxX: np array top-left (x,y) and w,h
        returns IoU
        """
        boxA = np.copy(boxAA) 
        boxB = np.copy(boxBB)
        boxA[2] = boxA[0] + boxA[2]
        boxB[2] = boxB[0] + boxB[2]
        boxA[3] = boxA[1] + boxA[3]
        boxB[3] = boxB[1] + boxB[3]
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

if __name__=='__main__':
    _=bb_intersection_over_union([14,116,163,298],[959,171,1271-959,562-171])
    print(_)