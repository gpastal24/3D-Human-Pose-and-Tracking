"""
    Multi cameras sample showing how to open multiple ZED in one program
"""

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

# tracker = TrackerSiamRPNBIG('test_SiamRPN/SiamRPNOTB.model')
# from tracker.byte_tracker import BYTETracker
#####COMMS####
# ERTZ = True
# if ERTZ == False:
#     host_name='localhost'
#     toggle_persistent=True.si
#     delivery_mode=2
#     queue_name='queue_name'
#     queue_durable=False
#     exchange_name='amq.topic'
#     exchange_type='topic'
#     exchange_durable=True
#     routing_key_name='amqpdemo.objects'
#     # ----------------------------- PRODUCER ---------------------#
#     connection = pika.BlockingConnection(pika.ConnectionParameters(host=host_name))
#     channel = connection.channel()
#     # declare exchange
#     channel.exchange_declare(exchange=exchange_name, exchange_type=exchange_type, durable=exchange_durable)
#     # declare queue
#     channel.queue_declare(queue=queue_name, durable=queue_durable, exclusive=False)
#     # create a binding between queue and exchange
#     channel.queue_bind(queue=queue_name, exchange=exchange_name, routing_key=routing_key_name)
# else:
# out_send = cv2.VideoWriter('appsrc  ! videoconvert ! x264enc tune=zerolatency ! rtph264pay ! udpsink host = 160.40.48.87 port=5000',cv2.CAP_GSTREAMER,0, 25, (640,360), True)

exchange_name = 'e1'  # 'server'
queue_name = 'q1'
queue_durable = True
exchange_durable = True
host_name =  '160.40.50.96'#'10.17.1.207'#0'160.40.50.96' #'10.42.0.66'#'160.40.50.97'#'160.40.48.223'
USER = 'test2'  # '10.42.0.1'
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
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')

from gi.repository import Gst, GstRtspServer, GObject

if True:
        class SensorFactory(GstRtspServer.RTSPMediaFactory):
            def __init__(self, **properties):
                super(SensorFactory, self).__init__(**properties)
                self.frame = np.zeros((640,480,3))
                self.number_frames = 0
                self.fps = 25
                self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
                self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                                    'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                                    '! videoconvert ! video/x-raw,format=I420 ' \
                                    '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                                    '! rtph264pay config-interval=1 name=pay0 pt=96' \
                                    .format(int(640), int(480), self.fps)
            # method to capture the video feed from the camera and push it to the
            # streaming buffer.
            def set(self, frame):
                self.frame = frame
                # print('frame set')

            def on_need_data(self, src, length):
                frame = cv2.resize(self.frame, (int(640), int(480)), \
                    interpolation = cv2.INTER_LINEAR)
                data = frame.tostring()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                retval = src.emit('push-buffer', buf)
                # print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
                                                                                #  self.duration,
                                                                                    # self.duration / Gst.SECOND))
                if retval != Gst.FlowReturn.OK:
                    print(retval)
            # attach the launch string to the override method
            def do_create_element(self, url):
                return Gst.parse_launch(self.launch_string)
            
            # attaching the source element to the rtsp media
            def do_configure(self, rtsp_media):
                self.number_frames = 0
                appsrc = rtsp_media.get_element().get_child_by_name('source')
                appsrc.connect('need-data', self.on_need_data)
                
        class GstServer(GstRtspServer.RTSPServer):
            def __init__(self, **properties):
                super(GstServer, self).__init__(**properties)
                self.factory = SensorFactory()
                self.factory.set_shared(True)
                self.set_service(str(8554))
                self.get_mount_points().add_factory('/video_stream', self.factory)
                self.attach(None)
            
            def set_frame(self, frame):
                self.factory.set(frame)
        
        class GstServer2(GstRtspServer.RTSPServer):
            def __init__(self, **properties):
                super(GstServer2, self).__init__(**properties)
                self.factory = SensorFactory()
                self.factory.set_shared(True)
                self.set_service(str(8555))
                self.get_mount_points().add_factory('/video_cv', self.factory)
                self.attach(None)
            
            def set_frame(self, frame):
                self.factory.set(frame)
        
        class GstServer3(GstRtspServer.RTSPServer):
            def __init__(self, **properties):
                super(GstServer3, self).__init__(**properties)
                self.factory = SensorFactory()
                self.factory.set_shared(True)
                self.set_service(str(8556))
                self.get_mount_points().add_factory('/video_stream', self.factory)
                self.attach(None)
            
            def set_frame(self, frame):
                self.factory.set(frame)
        class GstServer4(GstRtspServer.RTSPServer):
            def __init__(self, **properties):
                super(GstServer4, self).__init__(**properties)
                self.factory = SensorFactory()
                self.factory.set_shared(True)
                self.set_service(str(8557))
                self.get_mount_points().add_factory('/video_cv', self.factory)
                self.attach(None)
            
            def set_frame(self, frame):
                self.factory.set(frame)

        class GstServer5(GstRtspServer.RTSPServer):
            def __init__(self, **properties):
                super(GstServer5, self).__init__(**properties)
                self.factory = SensorFactory()
                self.factory.set_shared(True)
                self.set_service(str(8558))
                self.get_mount_points().add_factory('/video_stream', self.factory)
                self.attach(None)
            
            def set_frame(self, frame):
                self.factory.set(frame)

        class GstServer6(GstRtspServer.RTSPServer):
            def __init__(self, **properties):
                super(GstServer6, self).__init__(**properties)
                self.factory = SensorFactory()
                self.factory.set_shared(True)
                self.set_service(str(8559))
                self.get_mount_points().add_factory('/video_cv', self.factory)
                self.attach(None)
            
            def set_frame(self, frame):
                self.factory.set(frame)
        #Gst.init(None)
        #server = GstServer()
        #server2 = GstServer2()
        #server3 = GstServer3()
        #server4 = GstServer4()
        #server5 = GstServer5()
        #server6 = GstServer6()
        #loop = GObject.MainLoop()
        #loop2 = GObject.MainLoop()
        #loop3 = GObject.MainLoop()
        #loop4 = GObject.MainLoop()
        #loop5 = GObject.MainLoop()
        #loop6 = GObject.MainLoop()

        # loop.run()
        # loop2.run()
        #GObject.threads_init()
        #context = loop.get_context()
        #context2 = loop2.get_context()
        #context3 = loop3.get_context()
        #context4 = loop4.get_context()
        #context5 = loop5.get_context()
        #context6 = loop6.get_context()

# convert_to_trt(pose,'hrnet_trt.pth',256,192)
zed_list = []
left_list = []
depth_list = []
depth_images = []
timestamp_list = []
thread_list = []
pose_cap_list = []
processed_images = []
# new_data=[False,False]
stop_signal = False
d3_list = []
# from FasterVP.voxel_pose_class import VoxelMethod,parse_args
# from FasterVP.FasterVoxelPose.lib.core.config import config, update_config
# import torchvision.transforms as transforms
# import torch.backends.cudnn as cudnn

# args = parse_args()
# cfg = args.cfg
# update_config(cfg)
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
# print(VP.model.backbone.final_layer.weight)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
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

def get_pose_frame(index):
    # global xyz
    global new_data
    global image_c
    global image_d
    global d_image
    global left_list
    global head_pose_list
    # global np_image_c1
    global processed_images
    global zed_list
    global depth_list
    global depth_images
    global new_data
    global cam_info_list
    global frames_true
    global xyz_list
    global processed_3d
    global R
    global T
    global tracker_list
    # global args_track
    global t_ids_list
    # global Rot1
    # global T1
    while not stop_signal:
        # print(index,new_data[index])
        # temp = [None] * len(zed_list)
        if new_data[index]:
            lock.acquire()

            np_image_c1 = image_c[index].copy()
            np_image_d1 = image_d[index].copy()
            depth_image = d_image[index].copy()
            new_data[index]=False
            # print(np_image_c1.shape)
            # xyz_list[index] = xyz.copy()
            # print(xyz_list[index] )
            np_image_c1 = VP.resize_trans(np_image_c1)
            # print(VP.resize_transform)
            np_image_tens = VP.apply_transform(np_image_c1)
            # print(np_image_tens.shape)
            processed_images[index] = np_image_tens
            frames_true[index] = True
            processed_3d[index] = True
            # frames_true[index] = True
            # print('camera')
            # print('processed persons:',len(xyz),index)
            lock.release()


        else:
            time.sleep(0.03)
    # return xyz,np_image_c1,depth_image
def get_3d_pose():
    global processed_3d
    global final_poses_3d
    global new_3d_poses
    while not stop_signal:
        tot_cams = len(processed_3d)
        proc_cams = sum(processed_3d)
        if proc_cams == tot_cams:
            # proc
            lock.acquire()
            processed_3d = [False]*tot_cams
            #img = image_c[0].copy()
            #img2 = image_c[1].copy()
            #img3 = image_c[2].copy()
            #img4 = image_c[3].copy()
            #imgs = [image_c[i] for in range(0,tot_cams)]
            all_input =[]
            
            #inp1 = processed_images[0].cuda()
            #inp2 = processed_images[1].cuda()
            #inp3 = processed_images[2].cuda()
            #inp4 = processed_images[3].cuda()
            #all_input.append(inp1)
            #all_input.append(inp2)
            #all_input.append(inp3)
            #all_input.append(inp4)
            # print(all_input.shape)
            all_input = [processed_images[i].cuda() for i in range(0,tot_cams)]
            all_input = torch.stack(all_input, dim=0)

            t1=time.time()
            with torch.no_grad():
                # lock.acquire()
                starter.record()
                final_poses, poses, proposal_centers, _, input_heatmap = VP.model(views=all_input.unsqueeze(0), meta={}, cameras=VP.cameras, resize_transform=resize_transform)
                ender.record()
        # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                logger.info("elapsed pose time : {}".format(curr_time))
                # lock.release()  
                # print(final_poses)
            # t2 =time.time()
            # print(t2-t1)
            final_poses_3d[0] = final_poses.cpu().numpy().copy()
            final_poses_3d[1] = proposal_centers.cpu().numpy().copy()
            del final_poses,poses,proposal_centers,_,input_heatmap
            new_3d_poses[0] = True
            lock.release()
        else:
            time.sleep(0.03)
def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo.camera_resolution.width
    _intrinsics.height = cameraInfo.camera_resolution.height
    _intrinsics.ppx = cameraInfo.calibration_parameters.left_cam.cx
    _intrinsics.ppy = cameraInfo.calibration_parameters.left_cam.cy
    _intrinsics.fx = cameraInfo.calibration_parameters.left_cam.fx
    _intrinsics.fy = cameraInfo.calibration_parameters.left_cam.fy
    # _intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = rs.distortion.none
    _intrinsics.coeffs = [i for i in cameraInfo.calibration_parameters.left_cam.disto]
    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
    # result[0]: right, result[1]: down, result[2]: forward
    return result[0], result[1], result[2]

def signal_handler(signal, frame):
    global stop_signal
    stop_signal=True
    time.sleep(0.5)
    exit()

def grab_run(index):
    global stop_signal
    global zed_list
    global timestamp_list
    global left_list
    global depth_list
    global image_c
    global image_d
    global d_image
    global new_data
    runtime = sl.RuntimeParameters()
    while not stop_signal:
        err = zed_list[index].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            lock.acquire()
            zed_list[index].retrieve_image(left_list[index], sl.VIEW.LEFT)
            zed_list[index].retrieve_measure(depth_list[index], sl.MEASURE.DEPTH)
            zed_list[index].retrieve_image(depth_images[index], sl.VIEW.DEPTH)
            timestamp_list[index] = zed_list[index].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
            #lock.acquire()
            tmp = left_list[index].get_data().copy()
            image_c[index] = cv2.cvtColor(tmp, cv2.COLOR_BGRA2BGR)
            image_d[index] = depth_list[index].get_data().copy()
            d_image[index] = depth_images[index].get_data().copy()
            new_data[index] = True
            if index==0:
                img_send = image_c[index].copy()
                # out_send.write(cv2.resize(img_send,(640,360)))
            lock.release()

        time.sleep(0.03) #1ms
    zed_list[index].close()
	
def main():
    global stop_signal
    global zed_list
    global left_list
    global depth_list
    global timestamp_list
    global thread_list
    global image_c
    global image_d
    global d_image
    global new_data
    global Rot
    global R
    global T
    global pose_cap_list
    global processed_images
    global image_c
    global image_d
    global d_image
    global cam_info_list
    global frames_true
    global xyz_list
    global processed_3d
    global tracker_list
    global args_track
    global t_ids_list
    global args_lit
    global tracked_list
    global head_pose_list
    global final_poses_3d
    global new_3d_poses
    global messages_tracker
    global messages_tracker_truth
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

    signal.signal(signal.SIGINT, signal_handler)

    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode=sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units=sl.UNIT.METER# The framerate is lowered to avoid any USB3 bandwidth issues
    calib_serials = [22252043,27100523,26464933,27597034]
    R=[]
    T=[]
    #List and open cameras
    name_list = []
    last_ts_list = []
    cam_info_list = []
    cam_list_VP = []
    cameras = sl.Camera.get_device_list()
    index = 0
    rb_1 = c1()
    rb_2 = c2()
    rb_3 = c3()

    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        # print(cam.serial_number)
        if cam.serial_number in calib_serials:
            # print(cam.serial_number)

            indee = calib_serials.index(cam.serial_number)
            # temp_name_R = '/home/certh/EUROXR/calibration_zed/'+str(cam.serial_number)+'/R1.npy'
            # temp_name_T = '/home/certh/EUROXR/calibration_zed/'+str(cam.serial_number)+'/T1.npy'

            # temp_R = np.load(temp_name_R)
            # temp_T = np.load(temp_name_T)
            # # print(temp_name_R)
            # R.append(temp_R)
            # T.append(temp_T)
        name_list.append("ZED {}".format(cam.serial_number))
        cam_list_VP.append(cam.serial_number)
        logger.info("Opening {}".format(name_list[index]))
        # cam_info_list.append(cam.get_camera_information())
        zed_list.append(sl.Camera())
        left_list.append(sl.Mat())
        depth_list.append(sl.Mat())
        depth_images.append(sl.Mat())
        timestamp_list.append(0)
        last_ts_list.append(0)
        status = zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()
        index = index +1
    # out_vid = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))    #Start camera threads
    new_data = [False] * len(zed_list)
    image_c = [False] * len(zed_list)
    image_d = [False] * len(zed_list)
    d_image = [False] * len(zed_list)
    processed_images = [None] * len(zed_list)
    frames_true = [False] *  len(zed_list)
    processed_3d = [False] *  len(zed_list)
    tracker_list = [None]*len(zed_list)
    t_ids_list = [None]*len(zed_list)
    xyz_list = [None] * len(zed_list)
    # args_lit = [make_parser().parse_args()]*len(zed_list)
    tracked_list = [defaultdict()]*len(zed_list)
    head_pose_list = [None]*len(zed_list)
    # args_track = make_parser().parse_args()
    messages_tracker_truth = [False]*2
    messages_tracker = [None]*2
    tracker_3d =pose_3d_tracker_FVP(args.inside)
    data_saved = []

    from FasterVP.FasterVoxelPose.lib.utils.vis import save_debug_2d_images
    for index in range(0, len(zed_list)):
        cam_info_list.append(zed_list[index].get_camera_information())
        fps =30
        # if index ==0:
        #     args_lit[index].track_buffer =30
        # #     fps=29
        cx = cam_info_list[index].camera_configuration.calibration_parameters.left_cam.cx
        cy = cam_info_list[index].camera_configuration.calibration_parameters.left_cam.cy
        fx = cam_info_list[index].camera_configuration.calibration_parameters.left_cam.fx
        fy = cam_info_list[index].camera_configuration.calibration_parameters.left_cam.fy
        K_array = np.array([fx,fy,cx,cy])
        path = name_list[index]+'_intrinsics_HD.npy'
        np.save(path,K_array)
        # print(path)
        # tracker=byte_tracker.BYTETracker(args_lit[index],frame_rate=fps)
        # tracker_list[index]=tracker
        if zed_list[index].is_opened():
            thread_list.append(threading.Thread(target=grab_run, args=(index,)))
            thread_list[index].start()
            # print(zed_list[index])
            # print(index)
            
            pose_cap_list.append(threading.Thread(target=get_pose_frame,args=(index,)))
            pose_cap_list[index].start()
    final_poses_3d = [None,None] 
    new_3d_poses = [False]
    d3_list.append(threading.Thread(target=get_3d_pose, args=()))
    d3_list[0].start()
    d3_list.append(threading.Thread(target=rb_1.run))
    d3_list[1].start()
    d3_list.append(threading.Thread(target=rb_2.run))
    d3_list[2].start()
    d3_list.append(threading.Thread(target=rb_3.run))
    d3_list[3].start()
    # final_poses_3d = [None] 
    # new_3d_poses = [False]
     #Display camera images
    # print(t_ids_list)
    VP.cam_list = cam_list_VP
    VP.cameras = VP._get_cam(VP.cam_list)
    key = ''
    xyz1 = []
    xyz2 = []
    xyz3 = []
    times_list=[]
    counter =0
    individual_space_size = VP.individual_space_size
    space_size = VP.space_size

    space_size = space_size[:2]/10
    # init_tracker = True
    # start_track= False
    init_trackers = [True]*2
    start_trackers = [False]*2
    tracked_ids = [None]*2
    tracked_ids_keep = [None]*2
    tid_track = []
    trackers = [None]*2 #TrackerSiamRPNBIG('test_SiamRPN/SiamRPNOTB.model')

    # tracker = BYTETracker(make_parser().parse_known_args()[0],frame_rate=30)
    # fourcc = cv2.CV_FOURCC(*'DIVX')
    #out_vid = cv2.VideoWriter('output_clean.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (1280,720))    #Start camera threads
    #out_vid_2 = cv2.VideoWriter('output_clean2.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (1280,720))    #Start camera threads
    #out_vid_3 = cv2.VideoWriter('output_clean3.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (1280,720))    #Start camera threads


    while key != 113:  # for 'q' key



        for index in range(0, len(zed_list)):

            
            if zed_list[index].is_opened():

                    if frames_true[index]:
                       # lock.acquire()
                       frames_true[index] = False
                       # lock.release()
                        # cv2.imshow('color_pts'+name_list[index],processed_images[index])

                        # if index == 0 and processed_3d[index]==True:

                        #     xyz1 = xyz_list[0].copy()
                        #     # print(xyz1)
                        # elif index ==1 and processed_3d[index]==True:

                        #     xyz2 = xyz_list[1].copy()
        
        # if processed_3d[0] == True and processed_3d[1] == True and processed_3d[2]==True:
        if new_3d_poses[0] ==True:

            # counter+=1
            # if counter>615:
            #     counter=1
            # processed_3d[0] = False
            # processed_3d[1] = False
            lock.acquire()

            img = image_c[0].copy()
            img2 = image_c[1].copy()
            img3 = image_c[2].copy()
            # new_3d_poses[0]= False
            # lock.release()
            # lock.acquire()
            ######TEMP####

            counter+=1       
            final_poses = torch.from_numpy(final_poses_3d[0])
           
            prefix = '{}_{:08}'.format(os.path.join('outputs', 'validation'), counter)
            # img_file = save_debug_2d_images(config, {}, final_poses.cpu().numpy(), poses.cpu().numpy(), proposal_centers.cpu().numpy(), prefix)
            # img_3d = cv2.imread(img_file)
            # cv2.imshow('3d',img_3d)
            # print(img_3d.shape)
            # print(final_poses[0].shape)
            # print(final_poses)
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
            # points_proj = np.expand_dims(points_proj,axis=0)
            person_ids = np.arange(len(points_proj), dtype=np.int32)
            points_proj= points_proj[:len(pose_send)]
            # print(len(points_proj))
            # print(len(points3dcoco))
            bboxes_2d = boxes_from_poses(points_proj)
            bboxes_2d2 = boxes_from_poses(points_proj2)
            bboxes_2d3 = boxes_from_poses(points_proj3)
            img_ = img.copy()
            img2_ = img2.copy()
            img3_ = img3.copy()
            # print(cor_pose_order)
            # if cor_pose_order:
                # bbox_list = []
                # for i in range(len(pose_send)):
                #     bbox_list.append(bboxes_2d[i])
                # bbox_list = np.asarray(bbox_list)
                # bboxes_2d = bbox_list
                # cor_pose_order = np.asarray(cor_pose_order)
                # cor_pose_order
                # bboxes_2d = bboxes_2d[cor_pose_order.astype(int)]
            #print(start_trackers)
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
            #lock.release()
            #print(affs,t_ids)
            # method_frame, properties, body = channel_consum.basic_get(queue='cloud_track')
            # method_frame_del, properties, body_del = channel_consum.basic_get(queue='cloud_track_del')
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
                print(tid_del)
                messages_tracker_truth[1]=False
                # lock.release()
                if tid_del in tracked_ids_keep:

                    index_del = tracked_ids.index(tid_del)
                    tracked_ids[index_del] = None
                    tracked_ids_keep[index_del]=None
                    trackers[index_del] =None
                    start_trackers[index_del] = False
                    init_trackers[index_del] = True
                    print('tracker deleted!')
            #for i, (pt, pid,bbox_) in enumerate(zip(points_proj, t_ids,bboxes_2d)):
             #   img_ = draw_points_and_skeleton(img_, pt, joints_dict()['coco']['skeleton'], person_index=pid,
                #                                        points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10,confidence_threshold=0.4)
              #  org = (int(-20+pt[0][1]),int(pt[0][0]))#tuple(list(pt[i][:2]))
               # cv2.rectangle(img_,(bbox_[0],bbox_[1]),(bbox_[2],bbox_[3]),(255,255,0),2)

               # img_ = cv2.putText(img_, str(pid), org, cv2.FONT_HERSHEY_SIMPLEX, 1, 
                # (0,0,255), 2, cv2.LINE_AA, False)
            #for i, (pt, pid) in enumerate(zip(points_proj2, t_ids)):
             #   img2_ = draw_points_and_skeleton(img2_, pt, joints_dict()['coco']['skeleton'], person_index=pid,
               #                                         points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10,confidence_threshold=0.4)
              #  org = (int(-20+pt[0][1]),int(pt[0][0]))#tuple(list(pt[i][:2]))
              #  img2_ = cv2.putText(img2_, str(pid), org, cv2.FONT_HERSHEY_SIMPLEX, 1, 
              #   (0,0,255), 2, cv2.LINE_AA, False)
            #for i, (pt, pid) in enumerate(zip(points_proj3, t_ids)):
             #   img3_ = draw_points_and_skeleton(img3_, pt, joints_dict()['coco']['skeleton'], person_index=pid,
                 #                                       points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10,confidence_threshold=0.4)
             #   org = (int(-20+pt[0][1]),int(pt[0][0]))#tuple(list(pt[i][:2]))
              #  img3_ = cv2.putText(img3_, str(pid), org, cv2.FONT_HERSHEY_SIMPLEX, 1, 
              #   (0,0,255), 2, cv2.LINE_AA, False)
            # cv2.imshow("projected points",img_)
            # cv2.imshow("projected points2",img2_s
            # cv2.imshow("projected points3",img3)
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


            # out_vid.write(img)
            # out_vid_2.write(img2)
            # out_vid_3.write(img3)
           # print(t2-t1)
            # print(final_poses)
            # pose_last=fin_points_3c(xyz1,[],[],ids1,ids2,ids3,ids4,ids5,ids6)
            # pose_last = xyz1.copy()
            # pose_send = [f.pose for f in tracker_3d.f_ids]
            #print("~~~"*10)
           # print(len(pose_send))
            # print(pose_last)
            # time.sleep(0.5)
            # print(t_ids_list)
            # t_ids = person_ids = np.arange(len(points3dcoco), dtype=np.int32)
            # annos = create_json_rabbitmq(counter,t_ids, points3dcoco.tolist())
            tids_rev = [-i for i in t_ids]
            annos = create_json_rabbitmq(counter,t_ids, pose_send,affs)
            #annos_2 = create_json_rabbitmq(counter,tids_rev,pose_send,affs)
            # print(annos)
            # annos_2 = create_json_rabbitmq(counter,tids2_, xyz2.copy())
            # print()
            # print(annos_2)
            # print(annos_2[0])
            # print(annos)
            # out_act.write(cv2.resize(img,(640,360)))
            # out_act_2.write(cv2.resize(img2,(640,360)))
            # out_act_3.write(cv2.resize(img3,(640,360)))

            #server.set_frame(img)
            #server2.set_frame(img_)
            #context.iteration(False)
            #context2.iteration(False)

            #server3.set_frame(img2)
            #server4.set_frame(img2_)
            #context3.iteration(False)
            #context4.iteration(False)

            #server5.set_frame(img3)
            #server6.set_frame(img3_)
            #context5.iteration(False)
            #context6.iteration(False)         
            #save_path1 = 'cam1/'+str(counter).zfill(5)+'.jpeg'
            #save_path2 = 'cam2/'+str(counter).zfill(5)+'.jpeg'
            #save_path3 = 'cam3/'+str(counter).zfill(5)+'.jpeg'
            #cv2.imwrite(save_path1,img_)
            #cv2.imwrite(save_path2,img2_)
            #cv2.imwrite(save_path3,img3_)

            # annos = json.loads('jsonviewer.json')
            # producer_rabbitmq(channel, connection, annos = annos[0])[]
            if args.inside:
                message1= producer_rabbitmq(counter,channel, connection,queue_name = queue_name,exchange_name=exchange_name,routing_key_name=routing_key_name ,annos = annos[0])
            else:
                
                message2=producer_rabbitmq(counter,channel_2, connection,queue_name = queue_name_2,exchange_name=exchange_name_2,routing_key_name=routing_key_name_2 ,annos = annos[0])
            
            # js_name = 'json_outs_inside/'+str(counter).zfill(5)+'.json'
            # js_name_ = 'json_outs_out/'+str(counter).zfill(5)+'.json'

            # with open(js_name, 'w+') as j:
            #     # annos = json.loads(j.read())
            #     json.dump(message1.decode('utf-8'),j)
            # with open(js_name_, 'w+') as j:
            #     # annos = json.loads(j.read())
            #     json.dump(message2.decode('utf-8'),j)
            lock.release()
            #time.sleep(0.003)

            # print('message sent!')
            # xyz1=[]
            # xyz2 = []
            # lock.release()

        key = cv2.waitKey(1)
    # cv2.destroyAllWindows()

    #Stop the threads
    stop_signal = True
    for index in range(0, len(thread_list)):
        thread_list[index].join()
        pose_cap_list[index].join()
    d3_list[0].join()
    d3_list[1].join()
    d3_list[2].join()
    d3_list[3].join()
    rabbit_proc1.terminate()
    rabbit_proc4.terminate()
    print("\nFINISH")

if __name__ == "__main__":
    main()# -*- coding: utf-8 -*-

