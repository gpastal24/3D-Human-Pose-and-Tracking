#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:03:56 2022

@author: gpastal
"""

import torch 
from models_.hrnet import HRNet
from torchvision.transforms import transforms as TR
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
import cv2
import torch.nn.functional as F
import numpy as np
def pose_points_yolo5(detector,image,pose,Xw,Yw,Zw):
            
            # starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            transform = TR.Compose([
                TR.ToPILImage(),
                TR.Resize((256, 192)),  # (height, width)
                TR.ToTensor(),
                TR.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            detections = detector(image)
            # print(detections.shape)
            dets = detections.xyxy[0]
            dets = dets[dets[:,5] == 0.]
            dets = dets[dets[:,4] > 0.3]
            device='cuda:0'
            nof_people = len(dets) if dets is not None else 0
            boxes = np.empty((nof_people, 4), dtype=np.int32)
            images = torch.empty((nof_people, 3, 256, 192))  # (height, width)
            heatmaps = np.zeros((nof_people, 17, 64, 48),
                                dtype=np.float32)
            # starter.record()
            if dets is not None:
                for i, (x1, y1, x2, y2,_,_) in enumerate(dets):
                  # if i<1:
                    x1 = int(round(x1.item()))
                    x2 = int(round(x2.item())) # x1+int(round(x2))
                    y1 = int(round(y1.item()))
                    y2 = int(round(y2.item()))#int(round(y2))
                                

                    # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
                    correction_factor = 256 / 192 * (x2 - x1) / (y2 - y1)
                    if correction_factor > 1:
                        # increase y side
                        center = y1 + (y2 - y1) // 2
                        length = int(round((y2 - y1) * correction_factor))
                        y1 = max(0, center - length // 2)
                        y2 = min(image.shape[0], center + length // 2)
                    elif correction_factor < 1:
                        # increase x side
                        center = x1 + (x2 - x1) // 2
                        length = int(round((x2 - x1) * 1 / correction_factor))
                        x1 = max(0, center - length // 2)
                        x2 = min(image.shape[1], center + length // 2)
                    if x2>image.shape[1]:x2=image.shape[1]
                    if y2>image.shape[0]:y2=image.shape[0]
                    if y1<0: y1=0
                    if x1<0 : x1=0
                    # if x1
                    boxes[i] = [x1, y1, x2, y2]
                    
                    # print(boxes)
                    images[i] = transform(image[y1:y2, x1:x2, ::-1])
                    
            if images.shape[0] > 0:
                    images = images.to(device)
                    # boxes=boxes.to('cuda:1')
                    with torch.no_grad():
        
        
                                out = pose(images)
    
    
                          
                    # out = out.cpu().numpy()
                    # print(out.shape)
                    # out=out[0].unsqueeze(0)
                    pts = torch.empty((out.shape[0], out.shape[1], 3), dtype=torch.float32,device=device)
                    pts2 = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
                # For each human, for each joint: y, x, confidencet
                # time1=time.time()
                    # out=out.cpu.numpy()
                    (b,indices)=torch.max(out,dim=2)
                    (b,indices)=torch.max(b,dim=2)
                    
                    (c,indicesc)=torch.max(out,dim=3)
                    (c,indicesc)=torch.max(c,dim=2)
                    dim1= torch.tensor(1. / 64,device=device)
                    dim2= torch.tensor(1. / 48,device=device)
                    # dim1=1./64
                    # dim2=1/.48
                        # print(time33-time22)
                    for i in range(0,out.shape[0]):
                            # pt=cp.asarray(pt)
                            # print(time.time()-t333)
                            # pt=torch.cat((pt[0],pt[1]))
                            # print(pt)
                        # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                        # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                        # 2: confidences
                            pts[i, :, 0] = indicesc[i,:] * dim1 * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                            pts[i, :, 1] = indices[i,:] *dim2* (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                            pts[i, :, 2] = c[i,:]
                            # pts[i ,j ,2] = depth_img[int(pt_y),int(pt_x)]/1000.
                            
                            # pts2[i,j,0] = Xw[round(pt_y),round(pt_x)]
                            # pts2[i,j,1] = Yw[round(pt_y),round(pt_x)]
                        # pts2[i,j,2] = Zw[round(pt_y),round(pt_x)]
            # tim2=time.time()
            # print(1/(tim2-time1+0.000000001))
                    pts=pts.cpu().numpy()
                    # print(pts)
            else:
                pts = np.empty((0, 0, 3), dtype=np.float32)

            res = list()

            res.append(pts)
            
            # ender.record()
            # print(pts2)
            # curr_time = starter.elapsed_time(ender)/1000
            # torch.cuda.synchronize()
            # print(curr_time)

            if len(res) > 1:
                return res#,pts2
            else:
                return res[0]#,pts2
def pose_points_res18(image,poseboxes,poseres18):


            boxes = torch.empty((1, 4), dtype=torch.int32,device='cuda')
            images= torch.empty((1, 3, 256, 192),device='cuda') 
            
            transform = TR.Compose([
                    TR.ToPILImage(),
                    TR.Resize((256,192)),  # (height, width)
                    TR.ToTensor(),
                    TR.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
            # transform = torch.nn.Sequential(
                
            # transform=TR.Compose([TR.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            # heatmaps = torch.zeros((1, 17, 64, 48),
                                    # dtype=torch.float32).cuda()
    
            if poseboxes is not None and image is not None:
                # for  (x1, y1, x2, y2) in enumerate(poseboxes):
                        x1 = int(round(poseboxes[0]))
                        x2 = int(round(poseboxes[2]))
                        y1 = int(round(poseboxes[1]))
                        y2 = int(round(poseboxes[3]))
    
            # # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
                        correction_factor = 128 / 96 * (x2 - x1) / (y2 - y1)
                        if correction_factor > 1:
                            # increase y side
                            center = y1 + (y2 - y1) // 2
                            length = int(round((y2 - y1) * correction_factor))
                            y1 = max(0, center - length // 2)
                            y2 = min(image.shape[0], center + length // 2)
                        elif correction_factor < 1:
                            # increase x side
                            center = x1 + (x2 - x1) // 2
                            length = int(round((x2 - x1) * 1 / correction_factor))
                            x1 = max(0, center - length // 2)
                            x2 = min(image.shape[1], center + length // 2)
    
                        boxes = torch.tensor([x1,y1,x2,y2])
                        # print(boxes)
                        image = image[y1:y2, x1:x2, :: ]
                        
                        # image = image[::,y1:y2, x1:x2]
                        #img = image[y1:y2, x1:x2, ::-1] # works but lower performance
                        # cv2.imwrite('image.png', img) 
                        # img=np.transpose(img,(1,0,2))
            # image=image.unsqueeze(0)
            # image=image.type(torch.uint8)
            #print(image.shape)
            # image=image.transpose(0,1).transpose(1,2)
            # image=torch.from_numpy(image).cuda()
            # image=image.transpose(0,1).transpose(0,2)

            # image=image.to('cuda:0')
            # image=image.unsqueeze(0)
            # images[0]
            # image=F.interpolate(images,(256,192))#
            # image = image.unsqueeze(0)
            image = transform(image)
            image = image.unsqueeze(0)
            image = image.cuda()
            # with torch.no_grad():

            with torch.no_grad():
                # feats=backbones(image)
                # feats=feats[3]
 
                out=poseres18(image)
            

            pts=torch.empty((1,17,3),dtype=torch.float32,device='cuda')
            (b,indices)=torch.max(out,dim=2)
            (b,indices)=torch.max(b,dim=2)
            (c,indicesc)=torch.max(out,dim=3)
            (c,indicesc)=torch.max(c,dim=2)
            # c=c.cpu().numpy()
            # indicesc=indicesc.cpu().numpy()
            # indices=indices.cpu().numpy()
            dim1=torch.tensor(1./64,device='cuda')
            dim2=torch.tensor(1./48,device='cuda')
    
            pts[0, :, 0] = indicesc[0,:] *dim1 *(boxes[3] - boxes[1]) + boxes[1] # y values
            pts[0, :, 1] = indices[0,:] *dim2 * (boxes[2] - boxes[0]) + boxes[0] # x values
            pts[0, :, 2] = c[0,:]
            # pts=calculate_keypoints(out)
            pts=pts.cpu().numpy()

            return pts  
img = cv2.imread('Outputs/Captures/Camera1_img_00002.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = img[:,:,(2,1,0)]
pose=HRNet(32,17)
pose.load_state_dict(torch.load('./weights/pose_hrnet_w32_256x192.pth'))
pose.cuda().eval()
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s',_verbose=False)
# detector = torch.load('yolov5s.pt')
# detector = detector['model']
# pts = pose_points_res18(img, [150,50,420,371], pose)
pts = pose_points_yolo5(detector, img, pose, 0, 0, 0)
person_ids2 = np.arange(len(pts), dtype=np.int32)
for i, (pt, pid) in enumerate(zip(pts, person_ids2)):
    np_image_d1 = draw_points_and_skeleton(img, pt, joints_dict()['coco']['skeleton'], person_index=pid,
                                      points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10,confidence_threshold=0.4)  

