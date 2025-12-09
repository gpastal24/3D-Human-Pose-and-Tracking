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
from simpleHRNet.misc.visualization import (
    draw_points,
    draw_skeleton,
    draw_points_and_skeleton,
    joints_dict,
    check_video_rotation,
)
from pose_utils import pose_points_res18, pose_points, pose_points_yolo5
from general_utils import (
    jitter,
    jitter2,
    producer_rabbitmq,
    fix_head,
    create_json_rabbitmq,
    make_parser,
    check_centers,
    boxes_from_poses,
    filter_poses
)
from torch2trt import torch2trt, TRTModule
from new_functs_3cams import centers_3c, cross_id_3c, fin_points_3c, pose_3d_tracker_FVP
import pika
import json

# from tracker import byte_tracker
from collections import OrderedDict, defaultdict
import sys
from DARLENE_trackRPN.siamRPNBIG import TrackerSiamRPNBIG
from FasterVP.voxel_pose_class import VoxelMethod, parse_args
from FasterVP.FasterVoxelPose.lib.core.config import config, update_config
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torchvision.ops import box_iou
import multiprocessing as mp
from general_utils import set_message_multi, send_message_multi
import logging
from logger_helper import CustomFormatter
# mp.set_start_method('spawn')

logger_root = logging.getLogger()
logger_root.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger_root.addHandler(ch)
# logger_root.addHandler(ch)
logger = logging.getLogger("cv3d")
logging.getLogger("pika").setLevel(logging.WARNING)
args = parse_args()
cfg = args.cfg
update_config(cfg)
cam_tracking = args.tracking_cam
# def apply_transform(image):
#     if cfg.DATASET.COLOR_RGB:
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # print(self.transform)
#     # if cfg.transform:
#     #         image = self.transform(image)
#     return image
#         # all_input.append(input)
# def resize_trans(image):
#     image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
#     image=cv2.warpAffine(image, resize_transform, 
#                             (int(image_size[0]), int(image_size[1])),
#                             flags=cv2.INTER_LINEAR)
#     return image
def alter_bbox(box):
    # print(box)
    box[:,0] = box[:,0]/2
    box[:,1] = box[:,1]/2
    box[:,2]=(box[:,2]-box[:,0])/2#int((b[2] - b[0]) / 2),
    box[:,3]=(box[:,3]-box[:,1])/2#int((b[3] - b[1]) / 2),
    return box.astype(int)
if args.use_bytetrack:
    from tracker.byte_tracker import BYTETracker
    from general_utils import make_parser
    logger.warning(make_parser().parse_known_args()[0])
    bt_tracker = BYTETracker(make_parser().parse_known_args()[0],frame_rate=30)


if args.enable_rabbitcnc:
    exchange_name = "e1"  # 'server'
    queue_name = "q1"
    queue_durable = True
    exchange_durable = True
    host_name = args.cnc_host  #'160.40.50.96'#'10.17.1.207'#0'160.40.50.96' #'10.42.0.66'#'160.40.50.97'#'160.40.48.223'
    USER = args.rabbitcncuser  #'test2'  # '10.42.0.1'
    port = 5672
    credentials = pika.PlainCredentials(USER, USER)
    parameters = pika.ConnectionParameters(
        host=host_name, port=port, credentials=credentials
    )
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    # declare exchange
    exchange_type = "topic"
    routing_key_name = "b1"
    channel.exchange_declare(
        exchange=exchange_name, exchange_type=exchange_type, durable=exchange_durable
    )
    # declare queue
    channel.queue_declare(queue=queue_name, durable=queue_durable, exclusive=False)
    # create a binding between queue and exchange
    channel.queue_bind(
        queue=queue_name, exchange=exchange_name, routing_key=routing_key_name
    )

    exchange_name_2 = "e2"  # 'server'
    queue_name_2 = "q2"
    queue_durable = True
    exchange_durable = True
    # host_name = '10.42.0.177'  # '10.42.0.1'
    port = 5672
    credentials = pika.PlainCredentials(USER, USER)
    parameters = pika.ConnectionParameters(
        host=host_name, port=port, credentials=credentials
    )
    # connection = pika.BlockingConnection(parameters)
    channel_2 = connection.channel()
    # declare exchange
    exchange_type = "topic"
    routing_key_name_2 = "b2"
    channel_2.exchange_declare(
        exchange=exchange_name_2, exchange_type=exchange_type, durable=exchange_durable
    )
    # declare queue
    channel_2.queue_declare(queue=queue_name_2, durable=queue_durable, exclusive=False)
    # create a binding between queue and exchange
    channel_2.queue_bind(
        queue=queue_name_2, exchange=exchange_name_2, routing_key=routing_key_name_2
    )

    # exchange_name_consum = 'tracker'
    # queue_name_consum = 'cloud_track'
    # host_name_consum = '160.40.50.97'
    # port = 5672
    # credentials_consum = pika.PlainCredentials(USER,USER)
    # parameters_consum = pika.ConnectionParameters(host=host_name_consum, port=port, credentials=credentials_consum)
    # connection_consum = pika.BlockingConnection(parameters_consum)
    # channel_consum = connection_consum.channel()
    # exchange_type_consum='topic'
    routing_key_name_consum = "corrections.track"
    # out_act = cv2.VideoWriter('appsrc  ! videoconvert ! x264enc tune=zerolatency ! rtph264pay \
    #     ! udpsink host = 10.17.1.207 port=5000',cv2.CAP_GSTREAMER,0, 25, (640,360), True)
    # out_act_2 = cv2.VideoWriter('appsrc  ! videoconvert ! x264enc tune=zerolatency ! rtph264pay \
    #     ! udpsink host = 10.17.1.207 port=5001',cv2.CAP_GSTREAMER,0, 25, (640,360), True)
    # out_act_3 = cv2.VideoWriter('appsrc  ! videoconvert ! x264enc tune=zerolatency ! rtph264pay \
    #     ! udpsink host = 160.40.53.32 port=5000',cv2.CAP_GSTREAMER,0, 25, (640,360), True)
    # def callback(x1,x2,x3,body):
    #     # print (body.decode())
    #     # print(x2.exchange)
    #     global messages_tracker_truth
    #     global messages_tracker

    #     lock.acquire()

    #     # if x2.queue == 'cloud_track':
    #     data = json.loads(body)
    #     # print(data)
    #     messages[0] = [data['id'],data['rating']]

    #     # print(messages[1])
    #     message_truth[0] = True
    # # elif x2.queue=='cloud_track_del':
    #     data = json.loads(body)
    #     # print(data)
    #     messages[1] = data['id']
    #     # print(messages[1])
    #     message_truth[0] = True
    #     # print(message_truth)
    #     lock.release()

    def callback(x1, x2, x3, body):
        # print (body.decode())
        # print(x2.exchange)
        global messages_tracker_truth
        global messages_tracker

        lock.acquire()

        if x2.routing_key == "corrections.track":
            data = json.loads(body)
            # print(data)
            messages_tracker[0] = [data["id"], data["rating"]]
            # print(messages[1])
            messages_tracker_truth[0] = True
        elif x2.routing_key == "corrections.track_del":
            data = json.loads(body)
            # print(data)
            messages_tracker[1] = data["id"]
            # print(messages[1])
            messages_tracker_truth[1] = True
        elif x2.routing_key == "corrections.track_out":
            data = json.loads(body)
            # print(data)
            messages_tracker[0] = [data["id"], data["rating"]]
            # print(messages[1])
            messages_tracker_truth[0] = True
        elif x2.routing_key == "corrections.track_del_out":
            data = json.loads(body)
            # print(data)
            messages_tracker[1] = data["id"]
            # print(messages[1])
            messages_tracker_truth[1] = True
        # print(message_truth)
        lock.release()

    def callback_cloud(x1, x2, x3, body):
        global messages_tracker_truth
        global messages_tracker
        data = json.loads(body)
        print(data)
        lock.acquire()
        correction_list_in = [
            "corrections.in01",
            "corrections.in02",
            "corrections.in03",
        ]
        correction_list_out = [
            "corrections.out01",
            "corrections.out02",
            "corrections.out03",
        ]
        if args.inside:
            if x2.routing_key in correction_list_in:
                data = json.loads(body)
                messages_tracker[0] = [data["id"], data["rating"]]
                messages_tracker_truth[0] = True

        else:
            if x2.routing_key in correction_list_out:
                data = json.loads(body)
                messages_tracker[0] = [data["id"], data["rating"]]
                messages_tracker_truth[0] = True

        lock.release()

    if args.inside:

        class c1:
            def __init__(self):
                self.host_name = host_name  # '10.42.0.1'
                self.port = 5672
                self.credentials = pika.PlainCredentials(USER, USER)
                self.parameters = pika.ConnectionParameters(
                    host=self.host_name, port=self.port, credentials=self.credentials
                )
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()
                # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
                # self.channel = self.connection.channel()
                self.channel.exchange_declare(
                    exchange="tracker", durable=True, exchange_type="topic"
                )
                result = self.channel.queue_declare(
                    durable=True, queue="cloud_track"
                )  #'wecn01_3d')
                queue_name = result.method.queue
                # print(queue_name)
                binding_key = routing_key_name_consum  # "3d_proj_wecn01"

                # self.channel.queue_bind(exchange='tracker', queue=queue_name, routing_key=binding_key)
                # lock.acquire()
                self.channel.basic_qos(prefetch_count=10)
                self.channel.basic_consume(
                    queue=queue_name, on_message_callback=callback, auto_ack=True
                )
                # lock.release()

            def run(self):
                self.channel.start_consuming()
                time.sleep(0.10)

        class c2:
            def __init__(self):
                self.host_name = host_name  # '10.42.0.1'
                self.port = 5672
                self.credentials = pika.PlainCredentials(USER, USER)
                self.parameters = pika.ConnectionParameters(
                    host=self.host_name, port=self.port, credentials=self.credentials
                )
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()
                # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
                # self.channel = self.connection.channel()
                self.channel.exchange_declare(
                    exchange="tracker", durable=True, exchange_type="topic"
                )
                result = self.channel.queue_declare(
                    durable=True, queue="cloud_track_del"
                )  #'wecn01_3d')
                queue_name = result.method.queue
                # print(queue_name)
                binding_key = routing_key_name_consum  # "3d_proj_wecn01"
                self.channel.queue_bind(
                    exchange="tracker", queue=queue_name, routing_key=binding_key
                )
                # lock.acquire()
                self.channel.basic_qos(prefetch_count=10)
                self.channel.basic_consume(
                    queue=queue_name, on_message_callback=callback, auto_ack=True
                )
                # lock.release()

            def run(self):
                self.channel.start_consuming()
                time.sleep(0.10)

        class c3:
            def __init__(self):
                self.host_name = host_name  # '10.42.0.1'
                self.port = 5672
                self.credentials = pika.PlainCredentials(USER, USER)
                self.parameters = pika.ConnectionParameters(
                    host=self.host_name, port=self.port, credentials=self.credentials
                )
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()
                # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
                # self.channel = self.connection.channel()
                self.channel.exchange_declare(
                    exchange="command", durable=True, exchange_type="topic"
                )
                result = self.channel.queue_declare(
                    durable=True, queue="cloud3D_in"
                )  #'wecn01_3d')
                queue_name = result.method.queue
                # print(queue_name)
                binding_key = routing_key_name_consum  # "3d_proj_wecn01"
                self.channel.queue_bind(
                    exchange="command", queue=queue_name, routing_key="corrections.in01"
                )
                # lock.acquire()
                self.channel.basic_qos(prefetch_count=10)
                self.channel.basic_consume(
                    queue=queue_name, on_message_callback=callback_cloud, auto_ack=True
                )
                # lock.release()

            def run(self):
                self.channel.start_consuming()
                time.sleep(0.10)
    else:

        class c1:
            def __init__(self):
                self.host_name = host_name  # '10.42.0.1'
                self.port = 5672
                self.credentials = pika.PlainCredentials(USER, USER)
                self.parameters = pika.ConnectionParameters(
                    host=self.host_name, port=self.port, credentials=self.credentials
                )
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()
                # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
                # self.channel = self.connection.channel()
                self.channel.exchange_declare(
                    exchange="tracker", durable=True, exchange_type="topic"
                )
                result = self.channel.queue_declare(
                    durable=True, queue="cloud_track_out"
                )  #'wecn01_3d')
                queue_name = result.method.queue
                # print(queue_name)
                binding_key = routing_key_name_consum  # "3d_proj_wecn01"
                self.channel.queue_bind(
                    exchange="tracker",
                    queue=queue_name,
                    routing_key="corrections.track_out",
                )
                # lock.acquire()
                self.channel.basic_qos(prefetch_count=10)
                self.channel.basic_consume(
                    queue=queue_name, on_message_callback=callback, auto_ack=True
                )
                # lock.release()

            def run(self):
                self.channel.start_consuming()
                time.sleep(0.10)

        class c2:
            def __init__(self):
                self.host_name = host_name  # '10.42.0.1'
                self.port = 5672
                self.credentials = pika.PlainCredentials(USER, USER)
                self.parameters = pika.ConnectionParameters(
                    host=self.host_name, port=self.port, credentials=self.credentials
                )
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()
                # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
                # self.channel = self.connection.channel()
                self.channel.exchange_declare(
                    exchange="tracker", durable=True, exchange_type="topic"
                )
                result = self.channel.queue_declare(
                    durable=True, queue="cloud_track_del_out"
                )  #'wecn01_3d')
                queue_name = result.method.queue
                # print(queue_name)
                binding_key = routing_key_name_consum  # "3d_proj_wecn01"
                self.channel.queue_bind(
                    exchange="tracker",
                    queue=queue_name,
                    routing_key="corrections.track_del_out",
                )
                # lock.acquire()
                self.channel.basic_qos(prefetch_count=10)
                self.channel.basic_consume(
                    queue=queue_name, on_message_callback=callback, auto_ack=True
                )
                # lock.release()

            def run(self):
                self.channel.start_consuming()
                time.sleep(0.10)

        class c3:
            def __init__(self):
                self.host_name = host_name  # '10.42.0.1'
                self.port = 5672
                self.credentials = pika.PlainCredentials(USER, USER)
                self.parameters = pika.ConnectionParameters(
                    host=self.host_name, port=self.port, credentials=self.credentials
                )
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()
                # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
                # self.channel = self.connection.channel()
                self.channel.exchange_declare(
                    exchange="command", durable=True, exchange_type="topic"
                )
                result = self.channel.queue_declare(
                    durable=True, queue="cloud3D_out"
                )  #'wecn01_3d')
                queue_name = result.method.queue
                # print(queue_name)
                binding_key = routing_key_name_consum  # "3d_proj_wecn01"
                self.channel.queue_bind(
                    exchange="command", queue=queue_name, routing_key="corrections.out1"
                )
                # lock.acquire()
                self.channel.basic_qos(prefetch_count=10)
                self.channel.basic_consume(
                    queue=queue_name, on_message_callback=callback_cloud, auto_ack=True
                )
                # lock.release()

            def run(self):
                self.channel.start_consuming()
                time.sleep(0.10)
if args.usb:
    def get_frames(frames):
        return [
            np.split(frame, 2, axis=1)[0] for frame in frames
                     ]
else:
    def get_frames(frames):
        return [
                    frame.copy() for frame in frames
                ] 

lock = threading.Lock()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformss = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)
VP = VoxelMethod(config, transform=transformss)


# print(VP.transform)
resize_transform = torch.as_tensor(
    VP.resize_transform, dtype=torch.float, device="cuda:0"
)
resize_transform_np = resize_transform.cpu().numpy()
def apply_transform(image):
    if config.DATASET.COLOR_RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(self.transform)
    
    image = transformss(image)
    return image
        # all_input.append(input)
def resize_trans(image):
    image_size = np.array(config.NETWORK.IMAGE_SIZE)
    image=cv2.warpAffine(image, resize_transform_np, 
                            (int(image_size[0]), int(image_size[1])),
                            flags=cv2.INTER_LINEAR)
    return image
cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED
VP.load_weights(
    "FasterVP/FasterVoxelPose/output/panoptic/voxelpose_50/jln64/panoptic_5_cams_allseq/model_best.pth.tar"
)
VP.model.cuda().eval()
# VP.model.backbone.init_weights( pretrained='/home/certh/workspace/moving_out/3d_cams/FasterVP/FasterVoxelPose/models/pose_resnet50_panoptic.pth.tar')


VP.model.backbone.cuda().eval()
VP.model.backbone = TRTModule()
VP.model.backbone.load_state_dict(torch.load("trt_backbone.pth"))
VP.model.backbone.eval()
VP.model.pose_net.center_net = TRTModule()
VP.model.joint_net.conv_net = TRTModule()

VP.model.pose_net.center_net.load_state_dict(torch.load("trt_center_net.pth"))
VP.model.pose_net.center_net.eval()

VP.model.joint_net.conv_net.load_state_dict(torch.load("trt_p2p.pth"))
VP.model.joint_net.conv_net.eval()


def change_affiliation(aff_final, tracking_ids, data_saved):
    # pass
    for i, v in enumerate(data_saved):
        # print(v)
        # print(v['id'])
        # print(v['rating'])
        if v[0] in tracking_ids:
            inde = tracking_ids.index(v[0])
            aff_final[inde] = v[1]
            # print('here')
    return aff_final, tracking_ids
def byte_track_fun(pose_dict,final_pose_cpu,props):
    bboxes=[]
    # for joint in final_pose_cpu.numpy():
    #     # print(joint[:,4].max())
    #     bbox = [joint[:,0].min()+2250.,joint[:,1].min()+1200.,\
    #             joint[:,0].max()-joint[:,0].min(),joint[:,1].max()-joint[:,1].min(),1]
    #     bbox[2] = bbox[0]+bbox[2]
    #     bbox[3] = bbox[1]+bbox[3]
    #     a =np.asarray(bbox)
    #     a[:4] = a[:4]/10.
    #     bboxes.append(a)

    for prop in props.numpy():

        bbox = [prop[0]-1000,prop[1]-1000,\
                prop[0]+1000,prop[1]+1000,1]
        a =np.asarray(bbox)
        a[:4] = a[:4]/10.
        bboxes.append(a)
        # scores.append(0.7)
    # print(np.asanyarray(bboxes))
    # print(space_size)
    online_targets = bt_tracker.update(np.asarray(bboxes),space_size,space_size)

    online_tlwhs = []
    online_ids = []
    online_scores = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        xyxy = VP.xyxy(tlwh)

        online_tlwhs.append(np.asarray(xyxy))
        
        online_ids.append(tid)
        online_scores.append(t.score)
    corr_list = []

    if online_tlwhs:
        iou = torchvision.ops.box_iou(torch.from_numpy(np.asarray(bboxes)[:,:4]),torch.from_numpy(np.asarray(online_tlwhs)))
        
        # print(iou)
        # max_val = iou[0].max().numpy()
        for ii in iou:
            # print(ii)
            max_val = ii.max().numpy()
        # print(max_val)
            if max_val>0.1:
                # tracked_id = res[ii].argmax().numpy()
                corr_list.append(ii.argmax().numpy())


    tids = [online_ids[ind] for ind in corr_list]
    # final_pose_cpu = filter_poses(pose_dict,final_pose_cpu,tids)
    return tids
def siamese_track():
    for el, (init_tracker, start_track) in enumerate(
        zip(init_trackers, start_trackers)
    ):
        if init_tracker and start_track:
            if tid_track[0] in t_ids:
                j = t_ids.index(tid_track[0])
                bbox = bboxes_2d[cam_tracking][j][:4]
                bbox = np.array(
                    [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                ).astype(int)
                cv2.rectangle(
                    frames[cam_tracking],
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    (255, 0, 0),
                    4,
                )
                # bbox = np.asarray(bboxes_2d[3][j][:4]).astype(int)

                trackers[el].init(frames[cam_tracking], bbox)
                init_trackers[el] = False
        elif not init_tracker and start_track:
            bbox_track = trackers[el].update(frames[cam_tracking]).astype(int)
            cv2.rectangle(
                frames[cam_tracking],
                (bbox_track[0], bbox_track[1]),
                (bbox_track[0] + bbox_track[2], bbox_track[1] + bbox_track[3]),
                (0, 0, 255),
                4,
            )
            bbox_track_comp = [
                [
                    bbox_track[0],
                    bbox_track[1],
                    bbox_track[0] + bbox_track[2],
                    bbox_track[1] + bbox_track[3],
                ]
            ]
            # tracked_id,iou = VP.associate_bboxes(bbox_track,bboxes_2d)
            if len(bboxes_2d[cam_tracking]) > 0:
                res = box_iou(
                    torch.from_numpy(np.asarray(bbox_track_comp)),
                    torch.from_numpy(np.asarray(bboxes_2d[cam_tracking])),
                )
                # if res.size>1:
                max_val = res[0].max().numpy()
                if max_val > 0.2:
                    tracked_id = res[0].argmax().numpy()
                else:
                    tracked_id = None
            else:
                tracked_id = None
            # print(tracked_id)
            if tracked_id is not None:
                try:
                    # print(tracked_id)
                    # id_to_change = t_ids[tracked_id]
                    t_ids_temp = [f.f_id for f in tracker_3d.f_ids]
                    id_to_change = t_ids_temp[tracked_id]

                    if id_to_change != tracked_ids[el]:
                        if tracked_ids[el] in t_ids_temp:
                            for indeee, value in enumerate(t_ids_temp):
                                if (
                                    indeee != tracked_id
                                    and value == tracked_ids[el]
                                ):
                                    # cor_ind = t_ids_temp[indeee]
                                    if args.inside:
                                        max_id = max(t_ids_temp)
                                        tracker_3d.f_ids[indeee].f_id = int(
                                            max_id + 1
                                        )
                                    else:
                                        max_id = min(t_ids_temp)
                                        tracker_3d.f_ids[indeee].f_id = int(
                                            max_id - 1
                                        )
                        tracker_3d.f_ids[tracked_id].f_id = tracked_ids[el]
                    for indeee, value in enumerate(t_ids_temp):
                        if indeee != tracked_id and value == tracked_ids[el]:
                            # cor_ind = t_ids_temp[indeee]
                            if args.inside:
                                max_id = max(t_ids_temp)
                                tracker_3d.f_ids[indeee].f_id = int(max_id + 1)
                            else:
                                max_id = min(t_ids_temp)
                                tracker_3d.f_ids[indeee].f_id = int(max_id - 1)
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
if args.visualize_debug:
    def visualize():
        imgs_drawn = frames.copy()#left_images.copy()
        for ind,img in enumerate(imgs_drawn):
        # print(ind)
            for i, (pt, pid,bbox_) in enumerate(zip(points_proj[ind], t_ids,bboxes_2d[ind])):
                #print(bbox_)
                img = draw_points_and_skeleton(img, pt, joints_dict()['coco']['skeleton'], person_index=pid,
                                            points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                            points_palette_samples=10, confidence_threshold=0.4)
                # if i==0:
                    # print(list(pt[i][:2]))
                org = (int(-20+pt[0][1]),int(pt[0][0]))#tuple(list(pt[i][:2]))
                # cv2.rectangle(img,(bbox_[0],bbox_[1]),(bbox_[2],bbox_[3]),(255,255,0),1)
                img = cv2.putText(img, str(pid), org, cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0,0,255), 2, cv2.LINE_AA, False)
                imgs_drawn[ind] = img
    
        cv2.imshow("Camera 1", cv2.resize(imgs_drawn[0],(640,360)))
        cv2.imshow("Camera 2", cv2.resize(imgs_drawn[1],(640,360)))
else:
    def visualize():
        pass

def mp_read_cams(args,left_images,all_input,):
    list_args = [args.video, args.stream, args.usb]

    if sum(list_args) == 0:
        raise Exception("Select either video, stream or usb inference")
    elif sum(list_args) > 1:
        raise Exception("You cannot select more than 1 inference mode")
    if args.video:
        import glob

        files = sorted(glob.glob("videos/*.mp4"))
        # print(files)
        cap_list = [cv2.VideoCapture(file) for file in files][: args.cams_n]
        sleep_time = 0.02
        # print(cap_list)
    elif args.stream:
        # pass
        ports = range(5000, 5000 + args.cams_n)
        cap_list = [
            cv2.VideoCapture(
                "udpsrc port={} ! application/x-rtp,payload=96,encoding-name=H264 !\
        rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink drop=1".format(
                    port
                ),
                cv2.CAP_GSTREAMER,
            )
            for port in ports
        ]
        sleep_time = 0.001
        # cap = cv2.VideoCapture("udpsrc port=5000 ! application/x-rtp,payload=96,encoding-name=H264 !\
        # rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink drop=1", cv2.CAP_GSTREAMER);
        # cap2 = cv2.VideoCapture("udpsrc port=5001 ! application/x-rtp,payload=96,encoding-name=H264 !\
        # rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink drop=1", cv2.CAP_GSTREAMER);
        # cap3 = cv2.VideoCapture("udpsrc port=5002 ! application/x-rtp,payload=96,encoding-name=H264 !\
        # rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink drop=1", cv2.CAP_GSTREAMER);
        # cap4 = cv2.VideoCapture("udpsrc port=5003 ! application/x-rtp,payload=96,encoding-name=H264 !\
        # rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink drop=1", cv2.CAP_GSTREAMER);
    elif args.usb:
        cams = range(0,10,2)
        cap_list = [c for c in cams][:args.cams_n]
        cap_list = [cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2) for cap in cap_list]
        cap_list = [cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) for cap in cap_list]
        sleep_time=0.001
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    else:
        raise Exception("Select either video, stream or usb inference")

    opened_caps = [cap.isOpened() for cap in cap_list]
    if sum(opened_caps)!=args.cams_n:
        exit(-1)
    while True:
        # print(cap_list)
        read = [c.read() for c in cap_list]
        # print(read)
        rets = [r[0] for r in read]
        frames = [r[1] for r in read]
        # if args.stream:
        #     frames = [cv2.resize(f,(1280,720)) for f in frames]
        if sum(rets) == args.cams_n:
            # t_start=time.time()
            frames_ = get_frames(frames)
            left_images.put(get_frames(frames_))
            proc_images = [resize_trans(l_image) for l_image in frames_]
            final_images = [apply_transform(img) for img in proc_images]
            all_input_ = [i.cuda() for i in final_images]
            all_input.put(torch.stack(all_input_, dim=0))

        time.sleep(sleep_time)
# def estimate_3d_pose(inputs,VP):
#     final_poses, poses, proposal_centers, _, input_heatmap = VP.model(views=inputs.unsqueeze(0), meta={}, cameras=VP.cameras, resize_transform=resize_transform)
if __name__ == "__main__":
    # COnnections
    if args.use_bytetrack:
        track_fun = byte_track_fun
    else:
        track_fun = siamese_track
    if args.enable_rabbitcnc:
        msgBuffer1 = mp.Queue()
        msgBuffer2 = mp.Queue()
        msgBuffer3 = mp.Queue()
        msgBufferSet1 = mp.Queue()
        msgBufferSet2 = mp.Queue()
        msgBufferSet3 = mp.Queue()
        rabbit_proc1 = p = mp.Process(
            target=set_message_multi,
            args=(
                msgBuffer1,
                msgBufferSet1,
            ),
        )
        rabbit_proc2 = p = mp.Process(
            target=set_message_multi,
            args=(
                msgBuffer2,
                msgBufferSet2,
            ),
        )
        rabbit_proc3 = p = mp.Process(
            target=set_message_multi,
            args=(
                msgBuffer3,
                msgBufferSet3,
            ),
        )
        rabbit_proc4 = p = mp.Process(
            target=send_message_multi,
            args=(
                msgBuffer1,
                args,
                0,
            ),
        )
        rabbit_proc5 = p = mp.Process(
            target=send_message_multi, args=(msgBuffer2, args, 1)
        )
        rabbit_proc6 = p = mp.Process(
            target=send_message_multi, args=(msgBuffer3, args, 2)
        )
        rabbit_proc1.start()
        rabbit_proc2.start()
        rabbit_proc3.start()
        rabbit_proc4.start()
        rabbit_proc5.start()
        rabbit_proc6.start()
        rb_1 = c1()
        rb_2 = c2()
        rb_3 = c3()
    # camera list
    cam_list_VP = []
    if args.usb:
        cameras = sl.Camera.get_device_list()
        for cam in cameras:
            cam_list_VP.append(cam.serial_number)
    if args.cam_list:
        cam_list_VP = args.cam_list
    else:
        cam_list_VP = [22252043,26464933,33367045, 27597034,27100523][:args.cams_n]

    logger.warning("Camera list : {}".format(cam_list_VP))

    messages_tracker_truth = [False] * 2
    messages_tracker = [None] * 2
    tracker_3d = pose_3d_tracker_FVP(args.inside)
    data_saved = []
    d3_list = []
    # d3_list.append(threading.Thread(target=get_3d_pose, args=()))
    # d3_list[0].start()
    if args.enable_rabbitcnc:
        d3_list.append(threading.Thread(target=rb_1.run))
        d3_list[0].start()
        d3_list.append(threading.Thread(target=rb_2.run))
        d3_list[1].start()
        d3_list.append(threading.Thread(target=rb_3.run))
        d3_list[2].start()
    VP.cam_list = cam_list_VP
    VP.cameras = VP._get_cam(VP.cam_list)
    # 2D Trackers
    init_trackers = [True] * 2
    start_trackers = [False] * 2
    tracked_ids = [None] * 2
    tracked_ids_keep = [None] * 2
    tid_track = []
    trackers = [None] * 2
    # Open the ZED camera
    list_args = [args.video, args.stream, args.usb]
    if sum(list_args) == 0:
        raise Exception("Select either video, stream or usb inference")
    elif sum(list_args) > 1:
        raise Exception("You cannot select more than 1 inference mode")

    tstart = time.time()
    space_size  = VP.space_size[:2]/10.
    individual_space_size = VP.individual_space_size
    counter = 0
    pose_dict = OrderedDict()
    ctx = mp.get_context('spawn')
    # q = ctx.Queue()
    left_images = ctx.Queue()
    all_input = ctx.Queue()
    # mp.set_start_method('spawn')
    read_cams = ctx.Process(target=mp_read_cams, args=(args, left_images,all_input,))
        # thread_list[0].start()
    read_cams.start()

    while True:
        if left_images.empty():
            # print(frameBuffer.qsize())
            time.sleep(0.01)
            continue
        t1 = time.time()
        if left_images.qsize()>3:
            for _ in range(left_images.qsize()-2):
                left_images.get()
                all_input.get()

        # Get a new frame from camera
        frames = left_images.get()
        # print(frames)
                # t_start=time.time()
            # left_images = get_frames(frames)
        # proc_images = [VP.resize_trans(l_image) for l_image in frames]
        # final_images = [VP.apply_transform(img) for img in proc_images]
        # all_input = [i.cuda() for i in final_images]
        # all_input = torch.stack(all_input, dim=0)
        all_input_ = all_input.get()
        # print(all_input_.shape)
        with torch.no_grad():
            # t_start=time.time()
            if counter%2==0:
                final_poses, poses, proposal_centers, _, input_heatmap = VP.model(
                    views=all_input_.unsqueeze(0),
                    meta={},
                    cameras=VP.cameras,
                    resize_transform=resize_transform,
                )
                del all_input_
                torch.cuda.synchronize()
            else:
                del all_input_
            # t_end = time.time()
        # msg = "3D Pose Latency : {}".format(1/(t_end-t_start))
        # logger.info(msg)
        mask = proposal_centers[:, :, 3] >= 0
        props = proposal_centers[mask].cpu()

        final_pose = final_poses[0, mask[0]]
        final_pose_cpu = final_pose.cpu()
        if args.use_bytetrack:
            if counter%2==0:
                t_ids = track_fun(pose_dict,final_pose_cpu,props)
            final_pose_cpu = filter_poses(pose_dict,final_pose_cpu,t_ids)
    
        counter += 1

        points_proj = [
            VP.project_3d(VP.cameras["test"][k], final_pose_cpu)
            for k in range(len(frames))
        ]
        # prefix = '{}_{:08}'.format(os.path.join('outputs', 'validation'), counter)

        points3dcoco, bboxes_3d = VP.points_3d_to_coco_cpu(final_pose_cpu)
        points3dcoco, bboxes_3d = VP.post_proc_3d(points3dcoco, bboxes_3d)
        pose_send = points3dcoco.copy()
        points_proj = [p[:, :, [1, 0, 2]] for p in points_proj]
        bboxes_2d = [boxes_from_poses(p) for p in points_proj]

        # siamese_track()
        ##siam track
        if not args.use_bytetrack:
            tracker_3d.assign_ids(points3dcoco)
            t_ids = [f.f_id for f in tracker_3d.f_ids]
            pose_send = [f.pose for f in tracker_3d.f_ids]
            track_fun()
            t_ids = [f.f_id for f in tracker_3d.f_ids]

        msg = "Frame {} tracked ids {}".format(counter,t_ids)
        if counter%args.log_freq==0:
            logger.info(msg)
        affs = ["suspect"] * len(t_ids)
        affs, t_ids = change_affiliation(affs, t_ids, data_saved)

        if messages_tracker_truth[0]:
            tid_track = [messages_tracker[0][0]]
            data_saved.append(messages_tracker[0].copy())

            messages_tracker_truth[0] = False
            # lock.release()
            if not args.use_bytetrack and tid_track[0] not in tracked_ids:
                try:
                    _ind = tracked_ids.index(None)
                    tracked_ids[_ind] = tid_track[0]
                    tracked_ids_keep[_ind] = tid_track[0]
                    trackers[_ind] = TrackerSiamRPNBIG(
                        "test_SiamRPN/SiamRPNOTB.model"
                    )

                    start_trackers[_ind] = True
                except Exception:
                    logger.warning("trackers limit reached")

        msg = "Frame {} tracked ids {}".format(counter,tracked_ids_keep)
        if counter%args.log_freq==0:
            logger.info(msg)
        if messages_tracker_truth[1]:

            tid_del = messages_tracker[1]  # data['id'
            # logger.warning(tid_del)
            messages_tracker_truth[1] = False
            # lock.release()
            if tid_del in tracked_ids_keep:
                index_del = tracked_ids.index(tid_del)
                tracked_ids[index_del] = None
                tracked_ids_keep[index_del] = None
                trackers[index_del] = None
                start_trackers[index_del] = False
                init_trackers[index_del] = True
                logger.warning("tracker deleted!")

        # for ran_var in range(1,5):
        #     points_proj[:,ran_var,:] = points_proj[:,0,:]
        #     points_proj2[:,ran_var,:] = points_proj2[:,0,:]
        #     points_proj3[:,ran_var,:] = points_proj3[:,0,:]
        # points_proj[:,:,:2] = points_proj[:,:,:2]/2
        # points_proj2[:,:,:2] = points_proj2[:,:,:2]/2
        # points_proj3[:,:,:2] = points_proj3[:,:,:2]/2
        # bboxes_sent1=[]
        # bboxes_sent2=[]
        
        bboxes_sent = [
            alter_bbox(np.asarray(b))
            
            # [
            #     int(b[0] / 2),
            #     int(b[1] / 2),
            #     int((b[2] - b[0]) / 2),
            #     int((b[3] - b[1]) / 2),
            # ]
            for b in bboxes_2d
        ]
        points_proj_sent = [
            p[:,:,:2]/2
            for p in points_proj
        ]
        # bboxes_sent2 = [[int(b[0]/2),int(b[1]/2),int((b[2]-b[0])/2),int((b[3]-b[1])/2)]  for b in bboxes_2d2]
        # bboxes_sent3 = [[int(b[0]/2),int(b[1]/2),int((b[2]-b[0])/2),int((b[3]-b[1])/2)]  for b in bboxes_2d3]
        # bboxes_sent2 = [bboxes_2d2[0],bboxes_2d2[1],bboxes_2d2[2]-bboxes_2d2[0],bboxes_2d2[3]-bboxes_2d2[1]]
        # bboxes_sent3 = [bboxes_2d3[0],bboxes_2d3[1],bboxes_2d3[2]-bboxes_2d3[0],bboxes_2d3[3]-bboxes_2d3[1]]
        # logger.info(bboxes_sent1)
        # logger.warning(bboxes_sent2)
        # logger.warning(bboxes_sent3)
        if args.enable_rabbitcnc:
            msgBufferSet1.put(
                [[], bboxes_sent[0].tolist(), points_proj_sent[0].tolist(), t_ids, affs, frames[0]]
            )
        # msgBufferSet2.put([[],bboxes_sent2,points_proj2.tolist(),t_ids,affs,img2])

        # msgBufferSet3.put([[],bboxes_sent3,points_proj3.tolist(),t_ids,affs,img3])

        tids_rev = [-i for i in t_ids]
        annos = create_json_rabbitmq(counter, t_ids, pose_send, affs)

        # annos = json.loads('jsonviewer.json')
        # producer_rabbitmq(channel, connection, annos = annos[0])[]
        if args.enable_rabbitcnc:
            if args.inside:
                message1 = producer_rabbitmq(
                    counter,
                    channel,
                    connection,
                    queue_name=queue_name,
                    exchange_name=exchange_name,
                    routing_key_name=routing_key_name,
                    annos=annos[0],
                )
            else:
                message2 = producer_rabbitmq(
                    counter,
                    channel_2,
                    connection,
                    queue_name=queue_name_2,
                    exchange_name=exchange_name_2,
                    routing_key_name=routing_key_name_2,
                    annos=annos[0],
                )
        visualize()
        key = cv2.waitKey(1)
        if key == ord("q"):
            logger.warning("Stopping")
            break

        # Display images
        # cv2.imshow("frame", frame)
        # cv2.imshow("right", left_right_image[0])
        # cv2.imshow("left", left_right_image[1])
        # if cv2.waitKey(30) >= 0 :
        #    break
    if args.enable_rabbitcnc:
        d3_list[0].join()
        d3_list[1].join()
        d3_list[2].join()
        d3_list[3].join()
        rabbit_proc1.terminate()
        rabbit_proc2.terminate()
        rabbit_proc3.terminate()
        rabbit_proc4.terminate()
        rabbit_proc5.terminate()
        rabbit_proc6.terminate()
    # read_cams.join()
    read_cams.terminate()
    exit("Exit!")
