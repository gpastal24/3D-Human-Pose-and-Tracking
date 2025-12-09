
import pyzed.sl as sl
import cv2
import numpy as np
import threading
import time
import signal
# import pyrealsense2 as rs
import os
from simpleHRNet.models_.hrnet import HRNet
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms as TR
from simpleHRNet.SimpleHRNet import SimpleHRNet
from simpleHRNet.misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from pose_utils import pose_points_res18,pose_points,pose_points_yolo5
from general_utils import jitter,jitter2,producer_rabbitmq,fix_head,create_json_rabbitmq,make_parser,check_centers,boxes_from_poses
from torch2trt import torch2trt,TRTModule
from new_functs_3cams import centers_3c,cross_id_3c,fin_points_3c,pose_3d_tracker_FVP
import pika
import json
# from tracker import byte_tracker
from collections import OrderedDict,defaultdict
import sys
from DARLENE_trackRPN.siamRPNBIG import TrackerSiamRPNBIG
from FasterVP.voxel_pose_class import VoxelMethod,parse_args
from FasterVP.FasterVoxelPose.lib.core.config import config, update_config
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torchvision.ops import box_iou
import multiprocessing as mp
from general_utils import set_message_multi,send_message_multi

args = parse_args()
cfg = args.cfg
update_config(cfg)
import logging 
from logger_helper import CustomFormatter
logger_root = logging.getLogger()
logger_root.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger_root.addHandler(ch)
# logger_root.addHandler(ch)
logger = logging.getLogger('cv3d')
logging.getLogger("pika").setLevel(logging.WARNING)
exchange_name = 'e1'  # 'server'
queue_name = 'q1'
queue_durable = True
exchange_durable = True
host_name =  args.cnc_host#'160.40.50.96'#'10.17.1.207'#0'160.40.50.96' #'10.42.0.66'#'160.40.50.97'#'160.40.48.223'
USER = args.rabbitcncuser#'test2'  # '10.42.0.1'
port = 5672
credentials = pika.PlainCredentials(USER, USER)
parameters = pika.ConnectionParameters(host=host_name, port=port, credentials=credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()
# declare exchange
exchange_type='topic'
routing_key_name='b1'
channel.exchange_declare(exchange=exchange_name, exchange_type=exchange_type, durable=exchange_durable)
# declare queue
channel.queue_declare(queue=queue_name, durable=queue_durable, exclusive=False)
# create a binding between queue and exchange
channel.queue_bind(queue=queue_name, exchange=exchange_name, routing_key=routing_key_name)
    
exchange_name_2 = 'e2'  # 'server'
queue_name_2 = 'q2'
queue_durable = True
exchange_durable = True
# host_name = '10.42.0.177'  # '10.42.0.1'
port = 5672
credentials = pika.PlainCredentials(USER, USER)
parameters = pika.ConnectionParameters(host=host_name, port=port, credentials=credentials)
#connection = pika.BlockingConnection(parameters)
channel_2 = connection.channel()
# declare exchange
exchange_type='topic'
routing_key_name_2='b2'
channel_2.exchange_declare(exchange=exchange_name_2, exchange_type=exchange_type, durable=exchange_durable)
# declare queue
channel_2.queue_declare(queue=queue_name_2, durable=queue_durable, exclusive=False)
# create a binding between queue and exchange
channel_2.queue_bind(queue=queue_name_2, exchange=exchange_name_2, routing_key=routing_key_name_2)

# exchange_name_consum = 'tracker'
# queue_name_consum = 'cloud_track'
# host_name_consum = '160.40.50.97'
# port = 5672
# credentials_consum = pika.PlainCredentials(USER,USER)
# parameters_consum = pika.ConnectionParameters(host=host_name_consum, port=port, credentials=credentials_consum)
# connection_consum = pika.BlockingConnection(parameters_consum)
# channel_consum = connection_consum.channel()
# exchange_type_consum='topic'
routing_key_name_consum='corrections.track'
# out_act = cv2.VideoWriter('appsrc  ! videoconvert ! x264enc tune=zerolatency ! rtph264pay \
#     ! udpsink host = 10.17.1.207 port=5000',cv2.CAP_GSTREAMER,0, 25, (640,360), True)
# out_act_2 = cv2.VideoWriter('appsrc  ! videoconvert ! x264enc tune=zerolatency ! rtph264pay \
#     ! udpsink host = 10.17.1.207 port=5001',cv2.CAP_GSTREAMER,0, 25, (640,360), True)
# out_act_3 = cv2.VideoWriter('appsrc  ! videoconvert ! x264enc tune=zerolatency ! rtph264pay \
#     ! udpsink host = 160.40.53.32 port=5000',cv2.CAP_GSTREAMER,0, 25, (640,360), True)
def callback(x1,x2,x3,body):
    # print (body.decode())
    # print(x2.exchange)
    global messages_tracker_truth
    global messages_tracker

    lock.acquire()

    # if x2.queue == 'cloud_track':
    data = json.loads(body) 
    # print(data)
    messages[0] = [data['id'],data['rating']]

    # print(messages[1])
    message_truth[0] = True
# elif x2.queue=='cloud_track_del':
    data = json.loads(body) 
    # print(data)
    messages[1] = data['id']
    # print(messages[1])
    message_truth[0] = True
    # print(message_truth)
    lock.release()

def callback(x1,x2,x3,body):
    # print (body.decode())
    # print(x2.exchange)
    global messages_tracker_truth
    global messages_tracker

    lock.acquire()
    
    if x2.routing_key == 'corrections.track' :
        data = json.loads(body) 
        # print(data)
        messages_tracker[0] = [data['id'],data['rating']]
        # print(messages[1])
        messages_tracker_truth[0] = True
    elif x2.routing_key=='corrections.track_del':
        data = json.loads(body) 
        # print(data)
        messages_tracker[1] = data['id']
        # print(messages[1])
        messages_tracker_truth[1] = True
    elif x2.routing_key == 'corrections.track_out':
        data = json.loads(body) 
        # print(data)
        messages_tracker[0] = [data['id'],data['rating']]
        # print(messages[1])
        messages_tracker_truth[0] = True
    elif x2.routing_key=='corrections.track_del_out':
        data = json.loads(body) 
        # print(data)
        messages_tracker[1] = data['id']
        # print(messages[1])
        messages_tracker_truth[1] = True
    # print(message_truth)
    lock.release()
def callback_cloud(x1,x2,x3,body):
    # print (body.decode())
    # print(x2.exchange)
    global messages_tracker_truth
    global messages_tracker
    data = json.loads(body)
    print(data)
    lock.acquire()
    correction_list_in = ['corrections.in01','corrections.in02','corrections.in03']
    correction_list_out = ['corrections.out01','corrections.out02','corrections.out03']
    if args.inside:

        if x2.routing_key in correction_list_in :
            data = json.loads(body)
        # print(data)
            messages_tracker[0] = [data['id'],data['rating']]
        # print(messages[1])
            messages_tracker_truth[0] = True
        #elif x2.routing_key=='corrections.track_del':
         #   data = json.loads(body)
        # print(data)
          #  messages_tracker[1] = [data['id'],data['rating']]
        # print(messages[1])
           # messages_tracker_truth[1] = True
    else:
        if x2.routing_key in correction_list_out:
            data = json.loads(body)
        # print(data)
            messages_tracker[0] = [data['id'],data['rating']]
        # print(messages[1])
            messages_tracker_truth[0] = True
    #elif x2.routing_key=='corrections.track_del_out':
        #data = json.loads(body)
        # print(data)
        #messages_tracker[1] = data['id']
        # print(messages[1])
        #messages_tracker_truth[1] = True
    # print(message_truth)
    lock.release()
if args.inside:
    class c1():
    
        def __init__(self):
            self.host_name = host_name  # '10.42.0.1'
            self.port = 5672
            self.credentials = pika.PlainCredentials(USER, USER)
            self.parameters = pika.ConnectionParameters(host=self.host_name, port=self.port, credentials=self.credentials)
            self.connection = pika.BlockingConnection(self.parameters)
            self.channel = self.connection.channel()
            # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            # self.channel = self.connection.channel()
            self.channel.exchange_declare(exchange='tracker', durable=True, exchange_type='topic')
            result = self.channel.queue_declare(durable=True, queue='cloud_track')#'wecn01_3d')
            queue_name = result.method.queue
            # print(queue_name)
            binding_key = routing_key_name_consum#"3d_proj_wecn01"
           
            #self.channel.queue_bind(exchange='tracker', queue=queue_name, routing_key=binding_key)
            # lock.acquire()
            self.channel.basic_qos(prefetch_count=10)
            self.channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
            # lock.release()

        def run(self):
            self.channel.start_consuming()
            time.sleep(0.10)
            
    class c2():
    
        def __init__(self):
            self.host_name = host_name  # '10.42.0.1'
            self.port = 5672
            self.credentials = pika.PlainCredentials(USER, USER)
            self.parameters = pika.ConnectionParameters(host=self.host_name, port=self.port, credentials=self.credentials)
            self.connection = pika.BlockingConnection(self.parameters)
            self.channel = self.connection.channel()
            # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            # self.channel = self.connection.channel()
            self.channel.exchange_declare(exchange='tracker', durable=True, exchange_type='topic')
            result = self.channel.queue_declare(durable=True, queue='cloud_track_del')#'wecn01_3d')
            queue_name = result.method.queue
            # print(queue_name)
            binding_key = routing_key_name_consum#"3d_proj_wecn01"
            self.channel.queue_bind(exchange='tracker', queue=queue_name, routing_key=binding_key)
            # lock.acquire()
            self.channel.basic_qos(prefetch_count=10)
            self.channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
            # lock.release()

        def run(self):
            self.channel.start_consuming()
            time.sleep(0.10)
    class c3():

        def __init__(self):
            self.host_name = host_name  # '10.42.0.1'
            self.port = 5672
            self.credentials = pika.PlainCredentials(USER, USER)
            self.parameters = pika.ConnectionParameters(host=self.host_name, port=self.port, credentials=self.credentials)
            self.connection = pika.BlockingConnection(self.parameters)
            self.channel = self.connection.channel()
            # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            # self.channel = self.connection.channel()
            self.channel.exchange_declare(exchange='command', durable=True, exchange_type='topic')
            result = self.channel.queue_declare(durable=True, queue='cloud3D_in')#'wecn01_3d')
            queue_name = result.method.queue
            # print(queue_name)
            binding_key = routing_key_name_consum#"3d_proj_wecn01"
            self.channel.queue_bind(exchange='command', queue=queue_name, routing_key='corrections.in01')
            # lock.acquire()
            self.channel.basic_qos(prefetch_count=10)
            self.channel.basic_consume(queue=queue_name, on_message_callback=callback_cloud, auto_ack=True)
            # lock.release()

        def run(self):
            self.channel.start_consuming()
            time.sleep(0.10)
else:
    class c1():
    
        def __init__(self):
            self.host_name = host_name  # '10.42.0.1'
            self.port = 5672
            self.credentials = pika.PlainCredentials(USER, USER)
            self.parameters = pika.ConnectionParameters(host=self.host_name, port=self.port, credentials=self.credentials)
            self.connection = pika.BlockingConnection(self.parameters)
            self.channel = self.connection.channel()
            # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            # self.channel = self.connection.channel()
            self.channel.exchange_declare(exchange='tracker', durable=True, exchange_type='topic')
            result = self.channel.queue_declare(durable=True, queue='cloud_track_out')#'wecn01_3d')
            queue_name = result.method.queue
            # print(queue_name)
            binding_key = routing_key_name_consum#"3d_proj_wecn01"
            self.channel.queue_bind(exchange='tracker', queue=queue_name, routing_key='corrections.track_out')
            # lock.acquire()
            self.channel.basic_qos(prefetch_count=10)
            self.channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
            # lock.release()

        def run(self):
            self.channel.start_consuming()
            time.sleep(0.10)
            
    class c2():
    
        def __init__(self):
            self.host_name = host_name  # '10.42.0.1'
            self.port = 5672
            self.credentials = pika.PlainCredentials(USER, USER)
            self.parameters = pika.ConnectionParameters(host=self.host_name, port=self.port, credentials=self.credentials)
            self.connection = pika.BlockingConnection(self.parameters)
            self.channel = self.connection.channel()
            # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            # self.channel = self.connection.channel()
            self.channel.exchange_declare(exchange='tracker', durable=True, exchange_type='topic')
            result = self.channel.queue_declare(durable=True, queue='cloud_track_del_out')#'wecn01_3d')
            queue_name = result.method.queue
            # print(queue_name)
            binding_key = routing_key_name_consum#"3d_proj_wecn01"
            self.channel.queue_bind(exchange='tracker', queue=queue_name, routing_key='corrections.track_del_out')
            # lock.acquire()
            self.channel.basic_qos(prefetch_count=10)
            self.channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
            # lock.release()

        def run(self):
            self.channel.start_consuming()
            time.sleep(0.10)
    class c3():

        def __init__(self):
            self.host_name = host_name  # '10.42.0.1'
            self.port = 5672
            self.credentials = pika.PlainCredentials(USER, USER)
            self.parameters = pika.ConnectionParameters(host=self.host_name, port=self.port, credentials=self.credentials)
            self.connection = pika.BlockingConnection(self.parameters)
            self.channel = self.connection.channel()
            # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            # self.channel = self.connection.channel()
            self.channel.exchange_declare(exchange='command', durable=True, exchange_type='topic')
            result = self.channel.queue_declare(durable=True, queue='cloud3D_out')#'wecn01_3d')
            queue_name = result.method.queue
            # print(queue_name)
            binding_key = routing_key_name_consum#"3d_proj_wecn01"
            self.channel.queue_bind(exchange='command', queue=queue_name, routing_key='corrections.out1')
            # lock.acquire()
            self.channel.basic_qos(prefetch_count=10)
            self.channel.basic_consume(queue=queue_name, on_message_callback=callback_cloud, auto_ack=True)
            # lock.release()


        def run(self):
            self.channel.start_consuming()
            time.sleep(0.10)
lock = threading.Lock()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformss = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
VP=VoxelMethod(config,transform=transformss)
# print(VP.transform)
resize_transform = torch.as_tensor(VP.resize_transform,dtype=torch.float,device='cuda:0')
cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED
VP.load_weights("FasterVP/FasterVoxelPose/output/panoptic/voxelpose_50/jln64/panoptic_5_cams_allseq/model_best.pth.tar")
VP.model.cuda().eval()
# VP.model.backbone.init_weights( pretrained='/home/certh/workspace/moving_out/3d_cams/FasterVP/FasterVoxelPose/models/pose_resnet50_panoptic.pth.tar')


# VP.model.backbone.cuda().eval()
VP.model.backbone = TRTModule()
VP.model.backbone.load_state_dict(torch.load("trt_backbone.pth"))
VP.model.backbone.eval()
VP.model.pose_net.center_net = TRTModule()
VP.model.joint_net.conv_net = TRTModule()

VP.model.pose_net.center_net.load_state_dict(torch.load("trt_center_net.pth"))
VP.model.pose_net.center_net.eval()

VP.model.joint_net.conv_net.load_state_dict(torch.load("trt_p2p.pth"))
VP.model.joint_net.conv_net.eval()
def change_affiliation(aff_final,tracking_ids,data_saved):
    # pass  
    for i,v in enumerate(data_saved):
        #print(v)
        #print(v['id'])
        # print(v['rating'])
        if v[0] in tracking_ids:

            inde = tracking_ids.index(v[0])
            aff_final[inde]= v[1]
            # print('here')
    return aff_final,tracking_ids
# def estimate_3d_pose(inputs,VP):
#     final_poses, poses, proposal_centers, _, input_heatmap = VP.model(views=inputs.unsqueeze(0), meta={}, cameras=VP.cameras, resize_transform=resize_transform)
if __name__ == '__main__':

#COnnections
    msgBuffer1=mp.Queue()
    msgBuffer2=mp.Queue()
    msgBuffer3=mp.Queue()
    msgBufferSet1=mp.Queue()
    msgBufferSet2=mp.Queue()
    msgBufferSet3=mp.Queue()
    rabbit_proc1 = p = mp.Process(target=set_message_multi, args=( msgBuffer1, msgBufferSet1,))
   # rabbit_proc2 = p = mp.Process(target=set_message_multi, args=( msgBuffer2, msgBufferSet2,))
   # rabbit_proc3 = p = mp.Process(target=set_message_multi, args=( msgBuffer3, msgBufferSet3,))
    rabbit_proc4 = p = mp.Process(target=send_message_multi, args=( msgBuffer1,args,0,))
   # rabbit_proc5 = p = mp.Process(target=send_message_multi, args=( msgBuffer2, args,1))
   # rabbit_proc6 = p = mp.Process(target=send_message_multi, args=( msgBuffer3,args,2))
    rabbit_proc1.start()
    #rabbit_proc2.start()
    #rabbit_proc3.start()
    rabbit_proc4.start()
    #rabbit_proc5.start()
    #rabbit_proc6.start()
    rb_1 = c1()
    rb_2 = c2()
    rb_3 = c3()
#camera list
    cam_list_VP=[]
    cameras = sl.Camera.get_device_list()
    for cam in cameras :
        cam_list_VP.append(cam.serial_number)
    logger.warning("Camera list : {}".format(cam_list_VP))
    # cam_list_VP = [22252043,27100523,26464933,27597034]
    messages_tracker_truth = [False]*2
    messages_tracker = [None]*2
    tracker_3d =pose_3d_tracker_FVP(args.inside)
    data_saved = []
    d3_list =[]
    # d3_list.append(threading.Thread(target=get_3d_pose, args=()))
    # d3_list[0].start()
    d3_list.append(threading.Thread(target=rb_1.run))
    d3_list[0].start()
    d3_list.append(threading.Thread(target=rb_2.run))
    d3_list[1].start()
    d3_list.append(threading.Thread(target=rb_3.run))
    d3_list[2].start()
    VP.cam_list = cam_list_VP
    VP.cameras = VP._get_cam(VP.cam_list)
# 2D Trackers
    init_trackers = [True]*2
    start_trackers = [False]*2
    tracked_ids = [None]*2
    tracked_ids_keep = [None]*2
    tid_track = []
    trackers = [None]*2 
# Open the ZED camera
    cap = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    cap3 = cv2.VideoCapture(4)
    cap4 = cv2.VideoCapture(6)

    if cap.isOpened() == 0 and cap2.isOpened() == 0 and cap3.isOpened() == 0 and cap4.isOpened() == 0:
        exit(-1)

    # Set the video resolution to HD720 (2560*720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2)
    cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap4.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2)
    cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_list = [cap,cap2,cap3,cap4]
    tstart=time.time()
    counter = 1
    while True :
        # Get a new frame from camera
        read = [c.read() for c in cap_list]
        # print(a)
        rets = [r[0] for r in read]
        frames = [r[1] for r in read]
        # retval, frame = cap.read()
        # retval2,frame2  = cap2.read()
        # retval3,frame3 = cap3.read()
        # retval4,frame4 = cap4.read()
        if sum(rets)==4:
            # print(time.time()-tstart)
            # tstart=time.time() 
        # Extract left and right images from side-by-side
            left_images = [np.split(frame, 2, axis=1)[0] for frame in frames]
            proc_images = [VP.resize_trans(l_image) for l_image in left_images]
                # print(VP.resize_transform)
            final_images = [VP.apply_transform(img) for img in proc_images]
            #logger.warning(len(final_images))
            all_input = [i.cuda() for i in final_images]
            all_input = torch.stack(all_input, dim=0)
            with torch.no_grad():
                final_poses, poses, proposal_centers, _, input_heatmap = VP.model(views=all_input.unsqueeze(0)
            , meta={}, cameras=VP.cameras, resize_transform=resize_transform)
            final_poses = final_poses.cpu()

            img = left_images[0].copy()
            img2 = left_images[1].copy()
            img3 = left_images[2].copy()
            # new_3d_poses[0]= False
            # lock.release()
            # lock.acquire()
            ######TEMP####

            counter+=1       
            # final_poses = torch.from_numpy(final_poses_3d[0])
           
            prefix = '{}_{:08}'.format(os.path.join('outputs', 'validation'), counter)

            points_proj=VP.project_3d(VP.cameras['test'][0],final_poses[0])
            points_proj2=VP.project_3d(VP.cameras['test'][1],final_poses[0])
            points_proj3=VP.project_3d(VP.cameras['test'][2],final_poses[0])
            points3dcoco,bboxes_3d = VP.points_3d_to_coco(final_poses[0])
            points3dcoco,bboxes_3d = VP.post_proc_3d(points3dcoco,bboxes_3d)
            tracker_3d.assign_ids(points3dcoco)
            t_ids = [f.f_id for f in tracker_3d.f_ids]
            pose_send = [f.pose for f in tracker_3d.f_ids]
            # print(pose_send)
            pose_cor_order = np.asarray(pose_send)
            cor_pose_order = VP.reorder_poses(pose_cor_order,points3dcoco)
            #print(cor_pose_order)

            dummy=points_proj.copy()
            dummy[:,:,0] = points_proj[:,:,1]
            dummy[:,:,1] = points_proj[:,:,0]
            points_proj=dummy
            dummy=points_proj2.copy()
            dummy[:,:,0] = points_proj2[:,:,1]
            dummy[:,:,1] = points_proj2[:,:,0]
            points_proj2=dummy
            dummy=points_proj3.copy()
            dummy[:,:,0] = points_proj3[:,:,1]
            dummy[:,:,1] = points_proj3[:,:,0]
            points_proj3=dummy
            person_ids = np.arange(len(points_proj), dtype=np.int32)
            points_proj= points_proj[:len(pose_send)]

            bboxes_2d = boxes_from_poses(points_proj)
            bboxes_2d2 = boxes_from_poses(points_proj2)
            bboxes_2d3 = boxes_from_poses(points_proj3)
            img_ = img.copy()
            img2_ = img2.copy()
            img3_ = img3.copy()

            for el,(init_tracker,start_track) in enumerate(zip(init_trackers,start_trackers)):

                if init_tracker and start_track:
                    if tid_track[0] in t_ids:
                        j = t_ids.index(tid_track[0])
                        bbox = bboxes_2d[j][:4]
                        bbox = np.array([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]).astype(int)
                        trackers[el].init(img, bbox)
                        init_trackers[el] = False
                elif not init_tracker and start_track:
                    bbox_track = trackers[el].update(img).astype(int)
                    cv2.rectangle(img_,(bbox_track[0],bbox_track[1]),(bbox_track[0]+bbox_track[2],bbox_track[1]+bbox_track[3]),(0,0,255),4)
                    bbox_track_comp = [[bbox_track[0],bbox_track[1],bbox_track[0]+bbox_track[2],bbox_track[1]+bbox_track[3]]]
                    # tracked_id,iou = VP.associate_bboxes(bbox_track,bboxes_2d)
                    if len(bboxes_2d)>0:
                        res = box_iou(torch.from_numpy(np.asarray(bbox_track_comp)),torch.from_numpy(np.asarray(bboxes_2d)))
                        # if res.size>1:
                        max_val = res[0].max().numpy()
                        if max_val>0.2:
                            tracked_id = res[0].argmax().numpy()
                        else:
                            tracked_id = None
                    else:
                        tracked_id=None
                    # print(tracked_id)
                    if tracked_id is not None:
                        try:
                            # print(tracked_id)
                            # id_to_change = t_ids[tracked_id]
                            t_ids_temp = [f.f_id for f in tracker_3d.f_ids]
                            id_to_change = t_ids_temp[tracked_id]

                            if id_to_change!=tracked_ids[el]:
                                if tracked_ids[el] in t_ids_temp:
                                    for indeee, value in enumerate(t_ids_temp):
                                        if indeee!=tracked_id and value==tracked_ids[el]:
                                            # cor_ind = t_ids_temp[indeee]
                                            if args.inside:
                                                max_id = max(t_ids_temp)
                                                tracker_3d.f_ids[indeee].f_id = int(max_id+1)
                                            else:
                                                max_id = min(t_ids_temp)
                                                tracker_3d.f_ids[indeee].f_id = int(max_id-1)
                                tracker_3d.f_ids[tracked_id].f_id = tracked_ids[el]
                            for indeee,value in enumerate(t_ids_temp):
                                if indeee!=tracked_id and value==tracked_ids[el]:
                                    # cor_ind = t_ids_temp[indeee]
                                    if args.inside:
                                        max_id = max(t_ids_temp)
                                        tracker_3d.f_ids[indeee].f_id = int(max_id+1)
                                    else:
                                        max_id = min(t_ids_temp)
                                        tracker_3d.f_ids[indeee].f_id = int(max_id-1)
                        except Exception as e:
                            import traceback 
                            # print(res)
                            print(t_ids)
                            print(t_ids_temp)
                            # print(cor_ind)
                            print(tracked_id)
                            print(len(bboxes_2d))
                            # print(len(poin))
                            print([f.f_id for f in tracker_3d.f_ids])
                            print(len(points3dcoco))
                            traceback.print_exc()
                            
            t_ids =  [f.f_id for f in tracker_3d.f_ids]
            affs = ['suspect']*len(t_ids)

            affs,t_ids = change_affiliation(affs,t_ids,data_saved)

            if messages_tracker_truth[0] == True:

                tid_track=[messages_tracker[0][0]]
                data_saved.append(messages_tracker[0].copy())

                messages_tracker_truth[0] = False
                # lock.release()
                if tid_track[0] not in tracked_ids:
                    try:
                        _ind = tracked_ids.index(None)
                        tracked_ids[_ind]= tid_track[0]
                        tracked_ids_keep[_ind] = tid_track[0]
                        trackers[_ind] = TrackerSiamRPNBIG('test_SiamRPN/SiamRPNOTB.model')

                        start_trackers[_ind] = True
                    except:
                        logger.warning("trackers limit reached")
                # channel_consum.basic_ack(delivery_tag=method_frame.delivery_tag)
            #print(trackers)
            msg = "tracked ids {}".format(tracked_ids_keep)
            logger.info(msg)
            if messages_tracker_truth[1]==True:
                # lock.acquire()
            # if  is not None:
                # data = json.loads(body_del)
                # data_saved.append(data)
                tid_del=messages_tracker[1]#data['id'
                #logger.warning(tid_del)
                messages_tracker_truth[1]=False
                # lock.release()
                if tid_del in tracked_ids_keep:

                    index_del = tracked_ids.index(tid_del)
                    tracked_ids[index_del] = None
                    tracked_ids_keep[index_del]=None
                    trackers[index_del] =None
                    start_trackers[index_del] = False
                    init_trackers[index_del] = True
                    logger.warning('tracker deleted!')

            for ran_var in range(1,5):
                points_proj[:,ran_var,:] = points_proj[:,0,:]
                points_proj2[:,ran_var,:] = points_proj2[:,0,:]
                points_proj3[:,ran_var,:] = points_proj3[:,0,:]
            points_proj[:,:,:2] = points_proj[:,:,:2]/2
            points_proj2[:,:,:2] = points_proj2[:,:,:2]/2
            points_proj3[:,:,:2] = points_proj3[:,:,:2]/2
            #bboxes_sent1=[]
            #bboxes_sent2=[]
        
            bboxes_sent1 = [[int(b[0]/2),int(b[1]/2),int((b[2]-b[0])/2),int((b[3]-b[1])/2)] for b in bboxes_2d]
            bboxes_sent2 = [[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in bboxes_2d2]
            bboxes_sent3 = [[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in bboxes_2d3]
            #bboxes_sent2 = [bboxes_2d2[0],bboxes_2d2[1],bboxes_2d2[2]-bboxes_2d2[0],bboxes_2d2[3]-bboxes_2d2[1]]
            #bboxes_sent3 = [bboxes_2d3[0],bboxes_2d3[1],bboxes_2d3[2]-bboxes_2d3[0],bboxes_2d3[3]-bboxes_2d3[1]]
            msgBufferSet1.put([[],bboxes_sent1,points_proj.tolist(),t_ids,affs,img])
            #msgBufferSet2.put([[],bboxes_sent2,points_proj2.tolist(),t_ids,affs,img2])

            #msgBufferSet3.put([[],bboxes_sent3,points_proj3.tolist(),t_ids,affs,img3])



            tids_rev = [-i for i in t_ids]
            annos = create_json_rabbitmq(counter,t_ids, pose_send,affs)


            # annos = json.loads('jsonviewer.json')
            # producer_rabbitmq(channel, connection, annos = annos[0])[]
            if args.inside:
                message1= producer_rabbitmq(counter,channel, connection,queue_name = queue_name,exchange_name=exchange_name,routing_key_name=routing_key_name ,annos = annos[0])
            else:
                
                message2=producer_rabbitmq(counter,channel_2, connection,queue_name = queue_name_2,exchange_name=exchange_name_2,routing_key_name=routing_key_name_2 ,annos = annos[0])
            
            # left_image = numpy.split(frame, 2, axis=1)[0]
            # left_image2 = numpy.split(frame2, 2, axis=1)[0]
            # left_image3 = numpy.split(frame3, 2, axis=1)[0]
            # left_image4 = numpy.split(frame4, 2, axis=1)[0]

        # Display images
        #cv2.imshow("frame", frame)
        #cv2.imshow("right", left_right_image[0])
        #cv2.imshow("left", left_right_image[1])
        if cv2.waitKey(30) >= 0 :
            break
    d3_list[0].join()
    d3_list[1].join()
    d3_list[2].join()
    d3_list[3].join()
    rabbit_proc1.terminate()
    rabbit_proc4.terminate()
    exit(0)
