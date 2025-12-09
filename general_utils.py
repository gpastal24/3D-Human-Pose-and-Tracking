#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:49:22 2022

@author: gpastal
"""
import numpy as np
import pika
import json
from collections import OrderedDict
# from tracker import byte_tracker
import argparse
import cv2
import math
from copy import deepcopy
import time
import torch
import threading 
import sys


class consumer_class(threading.Thread):
    def __init__(self,host_name,USER,exchange,queue,callback,bind_key=None,*args,**kwargs):
        # threading.Thread.__init__(self)
        super(consumer_class, self).__init__(*args, **kwargs)
        self._kill = threading.Event()
        self._interval = 0.01
        self.host_name = host_name  # '10.42.0.1'
        self.port = 5672
        self.credentials = pika.PlainCredentials(USER, USER)
        self.parameters = pika.ConnectionParameters(
            host=self.host_name, port=self.port, credentials=self.credentials
        )
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()

        self.channel.exchange_declare(
            exchange=exchange, durable=True, exchange_type="topic"
        )
        # if bind_key is None: 
        result = self.channel.queue_declare(
                durable=True, queue=queue
            )  
        queue_name = result.method.queue
        if bind_key is not None:
            self.channel.queue_bind(
                exchange=exchange, queue=queue,routing_key = bind_key)
        self.channel.basic_qos(prefetch_count=10)
        self.channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True
        )
        self.consumer_tag = self.channel.consumer_tags
        # lock.release()
    def kill(self):
        self._kill.set()

    def run(self):
        # while True:
        while True:
            self.connection.process_data_events()
            is_killed = self._kill.wait(self._interval)
            if is_killed:
                break
        self.connection.close()
        return
    
    def stop(self):
        if self.is_alive():
            self.join()

    # def run(self):
    #     # while True:
    #     try:
    #         self.channel.start_consuming()
    #     except Exception as e:
    #         pass
    #     is_killed = self._kill.wait(self._interval)
    #     if is_killed:
    #         return
        
class NumpyArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NumpyArrayEncoder, self).default(obj)
# from apriltag import apriltag
def set_message_multi(msgBuffer, msgBufferSet):
    while True:
        if msgBufferSet.empty():
                    # print(frameBuffer.qsize())
            time.sleep(0.003)
            continue
        if msgBufferSet.qsize()>3:
            for _ in range(msgBufferSet.qsize()-2):
                msgBufferSet.get()

        a = 1
        #print(msgBuffer)
        polys_tr,boxes_tr,poses,tids,affiliations,image = msgBufferSet.get()
        
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),15]
        #print('before')
        result, imgencode = cv2.imencode('.jpg', cv2.resize(image,(640,360)), encode_param)
        #print('After')
        poses_trans = []
        poly_send = []

        annos = []
        annos.append({
        'tracking_id': tids,
        'keypoints': poses,
        'segmentation': polys_tr,
        'boxes': boxes_tr,
        'affiliation': affiliations,

        'image': imgencode
        })
        #print(annos)
        message = json.dumps(annos[0], cls=NumpyArrayEncoder).encode('ascii')
        msgBuffer.put(message)
def send_message_multi(msgBuffer,args,cam_num):
    if args.enable_rabbitcnc:
        cnc_credentials = pika.PlainCredentials(args.rabbitcncuser, args.rabbitcncpass)
        cnc_parameters = pika.ConnectionParameters(host=args.cnc_host,
                                                port=args.cnc_port, credentials=cnc_credentials)
        cnc_connection = pika.BlockingConnection(cnc_parameters)
        cnc_channel = cnc_connection.channel()

    while True:
        if msgBuffer.empty():
                    # print(frameBuffer.qsize())
            time.sleep(0.01)
            continue
        if msgBuffer.qsize()>3:
            for _ in range(msgBuffer.qsize()-2):
                msgBuffer.get()
        message =msgBuffer.get()
        #print(type(message))
        if args.inside:
            if cam_num==0:
                cnc_channel.basic_publish(exchange='video-exchange', routing_key='in01', body=message,properties=pika.BasicProperties(content_type='application/json',
                            delivery_mode=1, expiration='1000'))
            elif cam_num==1:
                cnc_channel.basic_publish(exchange='video-exchange', routing_key='in02', body=message,properties=pika.BasicProperties(content_type='application/json',
                        delivery_mode=1, expiration='1000'))
            elif cam_num==2:
                cnc_channel.basic_publish(exchange='video-exchange', routing_key='in03', body=message,properties=pika.BasicProperties(content_type='application/json',
                        delivery_mode=1, expiration='1000'))
            
            else:
                print('wrong cam number!')
        else:
            if cam_num==0:
                cnc_channel.basic_publish(exchange='video-exchange', routing_key='out01', body=message,properties=pika.BasicProperties(content_type='application/json',
                            delivery_mode=1, expiration='1000'))
            elif cam_num==1:
                cnc_channel.basic_publish(exchange='video-exchange', routing_key='out02', body=message,properties=pika.BasicProperties(content_type='application/json',
                        delivery_mode=1, expiration='1000'))
            elif cam_num==2:
                cnc_channel.basic_publish(exchange='video-exchange', routing_key='out03', body=message,properties=pika.BasicProperties(content_type='application/json',
                        delivery_mode=1, expiration='1000'))

            else:
                print('wrong cam number!')
def make_parser():
    parser = argparse.ArgumentParser("ByteTrack args")
   

    # parser.add_argument(
    #     "--path", default="scenario18.mp4", help="path to images or video"
    # )
    # parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")

    # parser.add_argument("--vid",default=True)

    # exp file


    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.2, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=180, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.99, help="matching threshold for tracking")
    # parser.add_argument(
    #     "--aspect_ratio_thresh", type=float, default=1.6,
    #     help="threshold for filtering out boxes of which aspect ratio are above the given value."
    # )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

def jitter(tracking,temp,id1):
    if str(id1) in tracking:
        if len(tracking[str(id1)]) > 5:
            tracking[str(id1)].pop(0)
            tracking[str(id1)].append(temp)
            return np.median(np.asarray(tracking[str(id1)]),axis=0)
        else:
            tracking[str(id1)].append(temp)
            return temp
    else:
        tracking[str(id1)]=[]
        tracking[str(id1)].append(temp)
        return temp
def jitter2(tracking,temp,id1)  :
    if str(id1) in tracking:
        if len(tracking[str(id1)]) > 5:
            tracking[str(id1)].pop(0)
            tracking[str(id1)].append(temp)
            return np.median(np.asarray(tracking[str(id1)]),axis=0)
        else:
            tracking[str(id1)].append(temp)
            return temp
    else:
        tracking[str(id1)]=[]
        tracking[str(id1)].append(temp)
        return temp
def filter_poses(t_dict,poses,tids) :
    for pose,idx in zip(poses,tids):
        if str(idx) in t_dict:
            if len(t_dict[str(idx)]) > 5:
                t_dict[str(idx)].pop(0)
                t_dict[str(idx)].append(pose)
            else:
                t_dict[str(idx)].append(pose)
        else:
            t_dict[str(idx)]=[]
            t_dict[str(idx)].append(pose)
    all_keys = t_dict.keys()
    tids_str = [str(key) for key in tids]
    del_keys = [k for k in all_keys if k not in tids_str]
    # print(del_keys)
    for key in del_keys:
        del t_dict[key] 
    poses_return = torch.empty(size=(len(tids),15,5),device=poses.device)
    for k,idx in enumerate(tids):
        # return torch.median()
        pos = t_dict[str(idx)]
        
        poses_return[k] = torch.Tensor(sum(pos)/len(pos))
    return poses_return    
def create_json_rabbitmq( FRAME_ID,tids,pose,affiliation,yaw=[]):
    '''
    Note: only 4 classes are being considered: person, backpack, handbag, suitcase.
    INPUT:
        1.FRAME_ID      : global (int) unique index of every frame
        2.cat_id        : (list) of class indices for each frame 
        3.detection_id  : (list) containing the indices of every detection in each frame
        4.tids          : (list) of indices unique for every tracked object among all frames
        5.pose          : (list) of numpy ndarrays
        6.polygon       : (list) of lists containing the polygons of every mask
    OUTPUT:
        1.annos         : (dictionary) holding all the info
    '''


    annos = []
    annos.append({
        'frame_id': FRAME_ID,
        # 'category_id': cat_id,
        # 'id': detection_id,
        'tracking_id': tids,
        'keypoints': pose,
        'affiliation' : affiliation,
        'yaw': yaw,
        # 'segmentation': polygon
    })
    return (annos)

def producer_rabbitmq(FRAME_ID,channel,
                      connection,
                      host_name='localhost',
                      toggle_persistent=True,
                      delivery_mode=2,
                      queue_name='queue_name',
                      queue_durable=True,
                      exchange_name='amq.topic',
                      exchange_type='topic',
                      exchange_durable=True,
                      routing_key_name='amqpdemo.objects',
                      annos=[],
                      ERTZ=False):
    '''
    INPUT:
        1. host_name        = [DEFAULT] 'localhost',
        2. toggle_persistent= [DEFAULT] True,
        3. delivery_mode    = [DEFAULT] 2,
        4. queue_name       = [DEFAULT] 'queue_name',
        5. queue_durable    = [DEFAULT] False,
        6. exchange_name    = [DEFAULT] 'wecn',
        7. exchange_type    = [DEFAULT] 'topic',
        8. exchange_durable = [DEFAULT] False,
        9. routing_key_name = [DEFAULT] 'wecn.instance_segmentation_json',
        10. annos           = [DEFAULT] []
        11. ERTZ            = [DEFAULT] False

    '''
    class NumpyArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NumpyArrayEncoder, self).default(obj)

    # global FRAME_ID

    if not ERTZ:
        message = json.dumps(annos, cls=NumpyArrayEncoder).encode('ascii')
        # print(message)
        # ------------------------- PUBLISH MESSAGE ------------------#
        if toggle_persistent:
            channel.basic_publish(exchange=exchange_name, routing_key=routing_key_name, body=message,
                                  properties=pika.BasicProperties(delivery_mode=delivery_mode, expiration='60'))
        else:
            channel.basic_publish(exchange=exchange_name, routing_key=routing_key_name, body=message)
#        print(message)
        #print("Message sent!")
        # ------------------- CLOSE CHANNEL AND CONNECTION -----------#
        # channel.close()
        # connection.close()
    else:
        message = json.dumps(annos[0], cls=NumpyArrayEncoder).encode('ascii')
        # ------------------------- PUBLISH MESSAGE ----------------------#
        if toggle_persistent:
            channel.basic_publish(exchange=exchange_name, routing_key=routing_key_name, body=message,
                                  properties=pika.BasicProperties(delivery_mode=delivery_mode, expiration='60'))
        else:
            channel.basic_publish(exchange=exchange_name, routing_key=routing_key_name, body=message,
                                  properties=pika.BasicProperties(expiration='60'))
        #print(message)
        #print("Message sent!")
        # ------------------- CLOSE CHANNEL AND CONNECTION -----------#
    return message
def fix_head(xyz):
    if xyz[0].any()!=0 and xyz[5].any()!=0 and xyz[6].any()!=0:
        # if np.linalg.norm(xyz[0][2]-xyz[5][2])>0.5 or np.linalg.norm(xyz[0][2]-xyz[6][2]) > 0.5:
            # print("head fixed")
            center_point=(xyz[5]+xyz[6])/2
            center_point[1]=center_point[1]+0.2
            xyz[0]=center_point
    if xyz[0].all()==0 and xyz[5].any()!=0 and xyz[6].any()!=0:
        # print("head fixed")

        center_point=(xyz[5]+xyz[6])/2
        center_point[1]=center_point[1]+0.2
        xyz[0]=center_point
    if xyz[7].all()==0 and xyz[5].any()!=0 and xyz[9].any()!=0:
        center_point=(xyz[5]+xyz[9])/2
        xyz[7]=center_point
    if xyz[8].all()==0 and xyz[6].any()!=0 and xyz[10].any()!=0:
        center_point=(xyz[6]+xyz[10])/2
        xyz[8]=center_point
        
    return xyz

def boxes_from_poses(pts):
    bboxes=[]
    for ind, i in enumerate(pts):
        # print(i)
        i = np.delete(i,obj=(1,2,3,4),axis=0)

        min_x = int(i[:,1].min())
        min_y = int(i[:,0].min())
        max_x = int(i[:,1].max())
        max_y = int(i[:,0].max())
        # print(min_x)
        bboxes.append([max(min_x-10,0),max(min_y-25,0),min(max_x+10,1280),min(max_y+20,720)])
    return bboxes
def pose_estimation_april_tag(april_detector,frame, matrix_coefficients, distortion_coefficients):

        '''
        frame - Frame from the video stream
        matrix_coefficients - Intrinsic matrix of the calibrated camera
        distortion_coefficients - Distortion coefficients associated with your camera
        return:-
        frame - The frame with the axis drawn on it
        '''

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
        # parameters = cv2.aruco.DetectorParameters_create()


        # corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        #     cameraMatrix=matrix_coefficients,
        #     distCoeff=distortion_coefficients)

            # If markers are detected
        # t1 = time.time()
        image_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = april_detector.detect(image_grey)
        # print(detections)s
        # print(detector)
        # t2 = time.time()
        # print('Latency == :',t2-t1)
        ids = np.array([i['id'] for i in detections], dtype=np.int32)[:, None]
        corners = [i['lb-rb-rt-lt'][None, ::-1].astype(np.float32) for i in detections]
        rvec=None
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.04, matrix_coefficients,
                                                                        distortion_coefficients)
                # Draw a square around the markers
                print(rvec*180/np.pi)
                # print(tvec)
                # cv2.aruco.drawDetectedMarkers(frame, corners) 

                # Draw Axis
                # cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

        return rvec

class CamParams(object):
    def __init__(self) -> None:
        self.fx=0
        self.fy=0
        self.cx=0
        self.cy=0
        self.tag_size=0
    def set_fx(self,fx):
        self.fx=fx
    def set_fy(self,fy):
        self.fy=fy
    def set_cx(self,cx):
        self.cx=cx
    def set_cy(self,cy):
        self.cy=cy
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def check_centers(center,xyz,tid,pose_confs):
    center_c = deepcopy(center)
    remov=[]
    for ind,i in enumerate(center_c):
        if i.all()==0. or np.isnan(i).all()==True:
            remov.append(ind)
    center = [v for i,v in enumerate(center) if i not in frozenset(remov)]
    xyz = [v for i,v in enumerate(xyz) if i not in frozenset(remov)]
    tid = [v for i,v in enumerate(tid) if i not in frozenset(remov)]
    pose_confs = [v for i,v in enumerate(pose_confs) if i not in frozenset(remov)]

    return center,xyz,tid,pose_confs

def merge_tids(tids1,tids2,ids1,ids2,ids3,ids4,ids5,ids6):
    id_counter = 0
    final_tids=[]
    for i in range(tids1):
        if i not in ids1:
            final_tids.append(tids1[i])
        else:
            if tids1[i]>=tids2[ids2[id_counter]]:
                final_tids.append(tids1[i])
            else:
                final_tids.append(tids2[ids2[id_counter]])
            id_counter+=1
    for i in range(tids2):
        if i not in ids2:
            final_tids.append(tids2[i])
    return final_tids

