#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:45:33 2022

@author: gpastal
"""
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms as TR
import numpy as np
import cv2
import logging
from  simpleHRNet.models_.hrnet import HRNet
from torch2trt import torch2trt,TRTModule
logger = logging.getLogger("Tracker !")
# from timerr import Timer
from pathlib import Path
import gdown
# timer_det = Timer()
# timer_track = Timer()
# timer_pose = Timer()
def pose_points_yolo5(detector,image,pose,tracker,args):
            # timer_det.tic()
            # starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            transform = TR.Compose([
                TR.ToPILImage(),
                TR.Resize((256, 192)),  # (height, width)
                TR.ToTensor(),
                TR.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            detections = detector.predict(image)
            # timer_det.toc()
            # logger.info('DET FPS -- %s',1./timer_det.average_time)
            # print(detections)
            # print(detections.shape)
            # dets = detections.xyxy[0]
            dets = detections
            if len(dets)>0:
            # print(dets)
                dets = dets[dets[:,5] == 0.]
                dets = dets[dets[:,4] > 0.3]
            # print(dets)
            # logger.warning(len(dets))
            if len(dets)>0:
                # timer_track.tic()
                online_targets=tracker.update(dets,[image.shape[0],image.shape[1]],image.shape)

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area :#:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # tracker.update()
                # timer_track.toc()
                # logger.info('TRACKING FPS --%s',1./timer_track.average_time)
                device='cuda'
                nof_people = len(dets) if dets is not None else 0
                # print(dets)
                # print(nof_people)
                boxes = torch.empty((nof_people, 4), dtype=torch.int32,device= 'cuda')
                # boxes = []
                images = torch.empty((nof_people, 3, 256, 192))  # (height, width)
                heatmaps = np.zeros((nof_people, 17, 64, 48),
                                    dtype=np.float32)
                # starter.record()
                # print(online_tlwhs)
                if online_tlwhs:
                    for i, (x1, y1, x2, y2) in enumerate(online_tlwhs):
                    # if i<1:
                        x1 = x1.astype(np.int32)
                        x2 = x1+x2.astype(np.int32)
                        y1 = y1.astype(np.int32)
                        y2 = y1+ y2.astype(np.int32)
                        
                        # print([x1,x2,y1,y2])
                        # image = cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,0), 1)
                # cv2.imwrite('saved.png',image)
                #         # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
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
                        # boxes[i]=[x1, y1, x2, y2]
                        # print(boxes[i])
                        # print(boxes)
                        # print(image.shape)
                        images[i] = transform(image[y1:y2, x1:x2, ::-1])
                        boxes[i]= torch.tensor([x1, y1, x2, y2])
                        
                if images.shape[0] > 0:
                        images = images.to(device)
                        # boxes=boxes.to('cuda:1')
                        # out = torch.empty((images.shape[0],17,64,48),device=device)
                        with torch.no_grad():
                            # for i in range(images.shape[0]):
                            # timer_pose.tic()
                            out = pose(images)
                            # timer_pose.toc()
                            # logger.info('POSE FPS -- %s',1./timer_pose.average_time)
        
                            
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
                        # print(dim1.dtype)
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
                                # print(boxes)
                                # print(online_tlwhs)
                                pts[i, :, 0] = indicesc[i,:] * dim1 * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                                pts[i, :, 1] = indices[i,:] *dim2* (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                                pts[i, :, 2] = c[i,:]
                                # print(pts)
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
                online_tlwhs = []
                online_ids = []
                online_scores=[]
            res = list()

            res.append(pts)
            
            # ender.record()
            # print(pts2)
            # curr_time = starter.elapsed_time(ender)/1000
            # torch.cuda.synchronize()
            # print(curr_time)

            if len(res) > 1:
                return res,online_tlwhs,online_ids,online_scores#,pts2
            else:
                return res[0],online_tlwhs,online_ids,online_scores#,pts2
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
def pose_points(detector,image,pose,Xw,Yw,Zw):
            # starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            transform = TR.Compose([
                TR.ToPILImage(),
                TR.Resize((256, 192)),  # (height, width)
                TR.ToTensor(),
                TR.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            detections = detector.predict_single(image)
            # print(detections.shape)
            device='cuda:0'
            nof_people = len(detections) if detections is not None else 0
            boxes = np.empty((nof_people, 4), dtype=np.int32)
            images = torch.empty((nof_people, 3, 256, 192))  # (height, width)
            heatmaps = np.zeros((nof_people, 17, 64, 48),
                                dtype=np.float32)
            # starter.record()
            if detections is not None:
                for i, (x1, y1, x2, y2,_,_) in enumerate(detections):
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
            image=image.to('cuda:0')
            image=image.unsqueeze(0)
            image=F.interpolate(image,(256,192))#
            image = transform(image)

            # with torch.no_grad():

            with torch.no_grad():
                # feats=backbones(image)
                # feats=feats[3]
 
                out=poseres18(image)
            

            pts=torch.empty((17,3),dtype=torch.float32,device='cuda')
            (b,indices)=torch.max(out,dim=2)
            (b,indices)=torch.max(b,dim=2)
            (c,indicesc)=torch.max(out,dim=3)
            (c,indicesc)=torch.max(c,dim=2)
            # c=c.cpu().numpy()
            # indicesc=indicesc.cpu().numpy()
            # indices=indices.cpu().numpy()
            dim1=torch.tensor(1./64,device='cuda')
            dim2=torch.tensor(1./48,device='cuda')
    
            pts[ :, 0] = indicesc[0,:] *dim1 *(boxes[3] - boxes[1]) + boxes[1] # y values
            pts[ :, 1] = indices[0,:] *dim2 * (boxes[2] - boxes[0]) + boxes[0] # x values
            pts[ :, 2] = c[0,:]
            # pts=calculate_keypoints(out)
            pts=pts.cpu().numpy()

            return pts  
        

def load_pose_trt():
    # pose=get_pose_net(8,False)
    # pose=TRTModule()
    MainModel = imp.load_source('MainModel', "./trt_int8/trt_int8/res18_pxl.py")
    MainModel = TRTModule()
    MainModel.load_state_dict(torch.load('./trt_int8/trt_int8/int8_res18.pth'))
    # MainModel.load_state_dict(torch.load('./trt_int8/trt_int8/fp16_res18.pth'))

    pose = MainModel.cuda().eval()
    # pose.load_state_dict(torch.load('./trt_int8/trt_int8/int8_res8.trt'))
    # pose.cuda().eval()
    return pose

def check_pose_weights():
    path = './trt_int8'
    if not Path(path).exists():


        Path(path).mkdir(parents=True, exist_ok=True)
        id =  '1NU0dEe3ATAxkII-Nhkpfv-LkWm7lOT3s'
        output = path+'/'+'trt_int8.zip'
        gdown.download(id=id,output=output, quiet=False, use_cookies=False)
        import zipfile
        with zipfile.ZipFile("./trt_int8/trt_int8.zip","r") as zip_ref:
            zip_ref.extractall("./trt_int8")

#    pose_file = 'res-8_check.pth'
#    file_path = path+'/'+pose_file
#    if not Path(file_path).is_file():
#        output = path+'/'+pose_file
#        id = '1dTPkqcnr4lK3cyprJtoVxg-W58xjn-Ns'
#        gdown.download(id=id,output=output,quiet=False,use_cookies=False)
    
        
        
def convert_to_trt(net,out,height,width):
    net.eval()     
    img = torch.rand(1,3,height,width).cuda()
    net_trt  =  torch2trt(net,[img],max_batch_size=1,fp16_mode=True)
    torch.save(net_trt.state_dict(),out)
    
def convert_pose():
    path1 = './trt_int8/trt_int8'
    file_path = path1+'/'+'int8_res18.pth'
    # file_path = path1+'/'+'fp16_res18.pth'
    if not Path(file_path).is_file():
        from torchvision.datasets import ImageFolder
        from torchvision.transforms import ToTensor, Compose, Normalize, Resize


        class ImageFolderCalibDataset():
        
            def __init__(self, root):
                self.dataset = ImageFolder(
                    root=root, 
                    transform=Compose([
        #                Resize((224, 224)),
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                )
        
            def __len__(self):
                return len(self.dataset)
        
            def __getitem__(self, idx):
                image, _ = self.dataset[idx]
                image = image[None, ...]  # add batch dimension
                return image
            
        dataset = ImageFolderCalibDataset('./trt_int8/trt_int8/calib_images')
        data = torch.rand(1,3,256,192).cuda()
        # MainModel = imp.load_source('MainModel', "./trt_int8/trt_int8/res18_pxl.py")
        # model = torch.load('./trt_int8/trt_int8/res18_pxl.pth').to(device='cuda').eval()
        model = HRNet(32,17)
        model.load_state_dict(torch.load('pose_hrnet_w32_256x192.pth'))
        model.cuda().eval()        
        model_trt = torch2trt(model, [data], int8_calib_dataset=dataset,int8_mode=True)
        # model_trt = torch2trt(model, [data],fp16_mode=True,max_batch_size=1)

        torch.save(model_trt.state_dict(), './int8_hrnet.pth')

    