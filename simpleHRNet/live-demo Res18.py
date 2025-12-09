import os
import sys
import argparse
import ast
import cv2
import time
import torch
# from vidgear.gears import CamGear
import numpy as np
import cupy as cp
sys.path.insert(1, os.getcwd())
# from cupy import deepcopy
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations
cp.cuda.Device(0).use()
import imp
from torchvision.transforms import transforms
def pose_points_res18(image,poseboxes):
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,192)),  # (height, width)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
            detections = model.detector.predict_single(image)
            net_time=time.time()
            # print(1/(net_time-t))
            nof_people = len(detections) if detections is not None else 0
            boxes = np.empty((nof_people, 4), dtype=np.int32)
            images = torch.empty((nof_people, 3, 256, 192))  # (height, width)
            heatmaps = np.zeros((nof_people, 17, 256 // 4, 192 // 4),
                                dtype=np.float32)

            if detections is not None:
                for i, (x1, y1, x2, y2, cls_conf, cls_pred) in enumerate(detections):
                    x1 = int(round(x1.item()))
                    x2 = int(round(x2.item()))
                    y1 = int(round(y1.item()))
                    y2 = int(round(y2.item()))

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

                    boxes[i] = [x1, y1, x2, y2]
                    images[i] = transform(image[y1:y2, x1:x2, ::-1])


        # print(pts)
            if images.shape[0] > 0: 
                images = images.to(device='cuda')
                with torch.no_grad():
                    out = poseres18(images)
                out = out.cpu().numpy()
            
            
                pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)    
                for i, human in enumerate(out):
                    heatmaps[i] = human
                    for j, joint in enumerate(human):
                        pt = np.unravel_index(np.argmax(joint), (64, 48))
                    # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                    # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                    # 2: confidences
                        pts[i, j, 0] = pt[0] * 1. / (256 // 4) * (boxes[3] - boxes[1]) + boxes[1]
                        pts[i, j, 1] = pt[1] * 1. / (192 // 4) * (boxes[2] - boxes[0]) + boxes[0]
                        pts[i, j, 2] = joint[pt]
            # tim2=time.time()
            # print(1/(tim2-time1+0.000000001))
            else :
                pts = np.empty((0, 0, 3), dtype=np.float32)
        # print('fps for pose =',(1/(time.time()-time1)))
            res = list()
            res.append(pts)
            return pts.asnumpy() 
def main(camera_id, filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution,
         single_person, use_tiny_yolo, disable_tracking, max_batch_size, disable_vidgear, save_video, video_format,
         video_framerate, device):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudd.benchmark=True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    def unravel_indices(
    indices: torch.LongTensor,
    shape: tuple):


        coord = []

        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = indices // dim

        coord = torch.stack(coord[::-1], dim=-1)

        return coord

    def unravel_index(
    indices: torch.LongTensor,
    shape: tuple,
):

        coord = unravel_indices(indices, shape)
        return tuple(coord)
    def pose_points_ress18(image):
            # t222=time.time()
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,192)),  # (height, width)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
            detections = model.detector.predict_single(image)
            net_time=time.time()
            # print(1/(net_time-t))
            nof_people = len(detections) if detections is not None else 0
            boxes = torch.empty((nof_people, 4), dtype=torch.float32).cuda()
            images = torch.empty((nof_people, 3, 256, 192))  # (height, width)


            if detections is not None:
                for i, (x1, y1, x2, y2, cls_conf, cls_pred) in enumerate(detections):
                    x1 = int(round(x1.item()))
                    x2 = int(round(x2.item()))
                    y1 = int(round(y1.item()))
                    y2 = int(round(y2.item()))

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
                    # box_list=[x1,y1,x2,y2]    
                    boxes[i] = torch.tensor([x1,y1,x2,y2])
                    images[i] = transform(image[y1:y2, x1:x2, ::-1])


            
            if images.shape[0] > 0: 
                images = images.to(device='cuda')
                with torch.no_grad():
                    # time1=time.time()
                    out = poseres18(images)
                    # time2=time.time()
                    # print(1/(time2-time1))
                # t1=time.time()    

                # out=out.cpu().numpy()
                # t2=time.time()
                # print(t2-t1)
                # # print(out.shape[1])
                # t3=time.time()
                pts = torch.empty((out.shape[0], out.shape[1], 3), dtype=torch.float32).cuda()  


                (b,indices)=torch.max(out,dim=2)
                (b,indices)=torch.max(b,dim=2)
                
                (c,indicesc)=torch.max(out,dim=3)
                (c,indicesc)=torch.max(c,dim=2)
                    # print(human)
                dim1= torch.tensor(1. / 64).cuda()
                dim2= torch.tensor(1. / 48).cuda()
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
            # tim2=time.time()
                        # print('time for computations=',time.time()-t333)
            # pr            if out:
                # del out
                # torch.cuda.empty_cache()
            else :
                pts = torch.empty((0, 0, 3), dtype=torch.float32)
        # print('fps for pose =',(1/(time.time()-time1)))
            # res = list()
            # res.append(pts)
            t4=time.time()
            # print('time for computations=',t4-time33)
            # print(t4-t3)
            pts=pts.cpu().numpy()
            # print(time.time()-t4)
            return pts

    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    if filename is not None:
        rotation_code = check_video_rotation(filename)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
    else:
        rotation_code = None
        if disable_vidgear:
            video = cv2.VideoCapture(camera_id)
            assert video.isOpened()
        else:
            video = CamGear(camera_id).start()

    if use_tiny_yolo:
         yolo_model_def="./models/detectors/yolo/config/yolov3-tiny.cfg"
         yolo_class_path="./models/detectors/yolo/data/coco.names"
         yolo_weights_path="./models/detectors/yolo/weights/yolov3-tiny.weights"
    else:
         yolo_model_def="./models/detectors/yolo/config/yolov3.cfg"
         yolo_class_path="./models/detectors/yolo/data/coco.names"
         yolo_weights_path="./models/detectors/yolo/weights/yolov3.weights"

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        return_bounding_boxes=not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device
    )
    MainModel = imp.load_source('MainModel', "C:\\Users\\giann\\res18_pxl1.py")
    poseres18=torch.load('C:\\Users\\giann\\res18_pxl.pth').to(device='cuda').eval()
    
    # print(model.model)
    if not disable_tracking:
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0

    while True:
        t = time.time()
        avg_fps=[]
        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
        else:
            frame = video.read()
            if frame is None:
                break
        frame = cv2.resize(frame,(640,360))
        time1=time.time()
        pts = pose_points_ress18(frame)
        time2=time.time()
        print("TIME FOR POSENET ==",(time2-time1))
        # print(pts)
        # pts=np.nan_to_num(pts,0.2)
        if not disable_tracking:
            boxes, pts = pts

        if not disable_tracking:
            if len(pts) > 0:
                if prev_pts is None and prev_person_ids is None:
                    person_ids = np.arange(next_person_id, len(pts) + next_person_id, dtype=np.int32)
                    next_person_id = len(pts) + 1
                else:
                    boxes, pts, person_ids = find_person_id_associations(
                        boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts, prev_person_ids=prev_person_ids,
                        next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4, smoothing_alpha=0.1,
                    )
                    next_person_id = max(next_person_id, np.max(person_ids) + 1)
            else:
                person_ids = np.array((), dtype=np.int32)

            prev_boxes = boxes.copy()
            prev_pts = pts.copy()
            prev_person_ids = person_ids

        else:
            print(".")
            person_ids = np.arange(len(pts), dtype=np.int32)

        for i, (pt, pid) in enumerate(zip(pts, person_ids)):
            frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=pid,
                                              points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                              points_palette_samples=10,confidence_threshold=0.5)
        detections = model.detector.predict_single(frame)
        fps = 1. / (time.time() - t+0.000001)
        print('\rframerate: %f fps' % fps, end='')
        avg_fps.append(fps)

        if has_display:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "fps=%s"%(fps), (1, 20), font, 0.75, (100, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('frame.png', frame)
            k = cv2.waitKey(1)
            if k == 27:  # Esc button
                if disable_vidgear:
                    video.release()
                    cv2.destroyAllWindows()
                else:
                    video.stop()
                break
                
        else:
            cv2.imwrite('frame.png', frame)

        if save_video:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
                video_writer = cv2.VideoWriter('output.avi', fourcc, video_framerate, (frame.shape[1], frame.shape[0]))
            video_writer.write(frame)

    if save_video:
        video_writer.release()
    cv2.destroyAllWindows()
    # "C:\\Users\\giann\\Desktop\\video_1.mp4"
    # Athens_1080p
    # video_path = "C:\\Users\\giann\\Desktop\\kinetics\\_\\-_q2PvFj5mk_000034_000044.mp4"
    # video_path = "C:\\Users\\giann\\Desktop\\kinetics\\_\\-_s_BQD-nSI_000644_000654.mp4"
    # video_path = "C:\\Users\\giann\\Desktop\\kinetics\\_\\0_k1lAjuusQ_000024_000034.mp4" 
    # video_path= "C:\\Users\\giann\\Desktop\\train\\Fight\\_2RYnSFPD_U_0.avi"
    # video_path= "C:\\Users\\giann\\Desktop\\train\\Fight\\_RziL1Ds6xU_0.avi"
    # video_path= "C:\\Users\\giann\\Desktop\\train\\Fight\\0lHQ2f0d_0.avi"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",default='videos/Athens_1080p.mp4'
                        ,type=str, )
    parser.add_argument("--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='PoseResNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=50)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str,default='weights/pose_resnet_50_256x192.pth')
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(256,192)')
    parser.add_argument("--single_person",default=False,
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--use_tiny_yolo",default=True,
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection). Ignored if --single_person",
                        action="store_true")
    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",default=True,
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=45)
    parser.add_argument("--disable_vidgear",default=True,
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--save_video", help="save output frames into a video.", action="store_true")
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                                     "See http://www.fourcc.org/codecs.php", type=str, default='MJPG')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default='cuda:0')
    args = parser.parse_args()
    main(**args.__dict__)
