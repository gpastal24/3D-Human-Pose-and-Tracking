import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
import numpy as np
import argparse
import os
from tqdm import tqdm
import cv2
import random
# import _init_paths
from .FasterVoxelPose.lib.core.config import config, update_config
from .FasterVoxelPose.lib.utils.utils import create_logger
from .FasterVoxelPose.lib.utils.transforms import get_affine_transform,get_scale
from .FasterVoxelPose.lib.utils.cameras import project_pose

from .FasterVoxelPose.lib.utils.vis import save_debug_2d_images
# import FasterVoxelPose.lib.dataset
from .FasterVoxelPose.lib.models.voxelpose import get_voxelpose
from scipy.spatial.transform import Rotation as rot

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--cnc_host',default='localhost')
    parser.add_argument('--rabbitcncuser',default='guest')
    parser.add_argument('--rabbitcncpass',default='guest')
    parser.add_argument('--cnc_port',default='5672')
    parser.add_argument('--exchange_stream',default='')
    parser.add_argument('--routing_key',default='test')
    parser.add_argument('--enable_rabbitcnc',default=False,type=bool)
    parser.add_argument('--cams_n',default=5,type=int)
    parser.add_argument('--usb',action='store_true')
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--stream',action='store_true')
    parser.add_argument('--tracking_cam',default=2,type=int)
    parser.add_argument('--use_bytetrack',action='store_true')
    parser.add_argument('--log_freq',default=100,type=int)
    parser.add_argument('--visualize_debug',action='store_true')
    parser.add_argument('--cam_list',default=[],type=list)
    parser.add_argument('--trt',action='store_true')
    parser.add_argument('--test_mode',action='store_true')
    parser.add_argument('--device',default = 'cuda:0')
    parser.add_argument('--save_video', action='store_true', help='Save output videos')
    args, _ = parser.parse_known_args()
    update_config(args.cfg)

    return args

class VoxelMethod(object):
    def __init__(self,cfg,device='cuda:0',transform=None):
        # super(t, obj)
        args=parse_args()
        update_config(args.cfg)
        # print(config)
        self.cfg = args.cfg
        self.root_id = cfg.DATASET.ROOTIDX
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE
        self.num_views = cfg.DATASET.CAMERA_NUM
        self.color_rgb = cfg.DATASET.COLOR_RGB
        self.num_joints = 15
        self.num_views = cfg.DATASET.CAMERA_NUM

        self.ori_image_width = cfg.DATASET.ORI_IMAGE_WIDTH
        self.ori_image_height = cfg.DATASET.ORI_IMAGE_HEIGHT

        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE)
        self.sigma = cfg.NETWORK.SIGMA
    
        self.space_size = np.array(cfg.CAPTURE_SPEC.SPACE_SIZE)
        self.space_center = np.array(cfg.CAPTURE_SPEC.SPACE_CENTER)
        self.voxels_per_axis = np.array(cfg.CAPTURE_SPEC.VOXELS_PER_AXIS)
        self.individual_space_size = np.array(cfg.INDIVIDUAL_SPEC.SPACE_SIZE)
        
        # self.root_id = cfg.DATASET.ROOTIDX
        self.resize_transform = self._get_resize_transform()
        self.sequence_list = ['test']
        self.transform = transform
        self.cam_list = [27100523,26464933,22252043]
        self.cam_list = [27100523,22252043,26464933]
        # # self.cam_list = [26464933,22252043,27100523]
        self.cam_list = [22252043,26464933,27100523]
        # # self.cam_list = [22252043]
        # self.cam_list = [26464933,22252043]
        # self.cam_list = [26464933]



        # self.cam_list = [26464933,27100523,22252043]


        self.model = get_voxelpose(config,is_train=False)
        self.device = device
        self.cameras = self._get_cam(self.cam_list)

    def load_weights(self,path):
        # print(params.keys())
        self.model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
    
    def apply_transform(self,image):
        if self.color_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(self.transform)
        if self.transform:
                image = self.transform(image)
        return image
            # all_input.append(input)
    def resize_trans(self,image):
        image=cv2.warpAffine(image, self.resize_transform, 
                                (int(self.image_size[0]), int(self.image_size[1])),
                                flags=cv2.INTER_LINEAR)
        return image
    def _get_resize_transform(self):
        r = 0
        c = np.array([self.ori_image_width / 2.0, self.ori_image_height / 2.0])
        s = get_scale((self.ori_image_width, self.ori_image_height), self.image_size)
        trans = get_affine_transform(c, s, r, self.image_size)
        # print(trans)
        return trans

    def _get_cam(self,cam_list):
        cameras = dict()
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0]])

        # M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0],[0.0, 1.0, 0.0]])
        for seq in self.sequence_list:
            cameras[seq] = []

            # cam_file = osp.join(self.dataset_root, seq, "calibration_{:s}.json".format(seq))
            # with open(cam_file, "r") as f:
            #     calib = json.load(f)

            # for cam in calib["cameras"]:
            #     if (cam['panel'], cam['node']) in self.cam_list:
            #         sel_cam = {}
            #         sel_cam['K'] = np.array(cam['K'])
            #         sel_cam['distCoef'] = np.array(cam['distCoef'])
            #         sel_cam['R'] = np.array(cam['R']).dot(M)
            #         sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
            # print(cam_list)
            print(seq)
            for cam in cam_list:
                cameras[seq].append({})
            
            # convert the format of camera parameters
            for k, v in enumerate(cam_list):
                our_cam = dict()
                path_k = str(v) +'/'+str(v)+'_intrinsics_HD1.npy'
                # path_k = str(v)+'_intrinsics.npy'

                path_r = str(v) +'/'+'R1_calib.npy'
                path_t = str(v) +'/'+'T1_calib.npy'
                K = np.load(path_k)
                R = np.load(path_r)
                R= R
                # print(R)
                # # print(Rot.from_matrix())
                # r = rot.from_matrix(R)
                # print(r.as_euler('xyz',degrees=False))
                # r_e = r.as_euler('xzy',degrees=False)
                # R = rot.from_euler('xyz',r_e)
                # print(R.as_matrix())
                # r_vec = r.as_euler('xyz',degrees=False)
                # r_vec = rot.from_rotvec(r_vec,'xzy')
                # print(r_vec.as_matrix())
                # print(r_vec)
                R = R.dot(M)
                # print(R.T)
                T = np.load(path_t)
                # T[2] = T[2]+1.41
                # T = T.transpose((0,2,1))
                T = T.reshape((3,1))
                # T[2] = T[2]+1.41
                # print(T)
                # T = T.transpose(0,2,1)
                # T_dum = T.copy()
                # T_dum[1] = T[2]
                # T_dum[2] = T[1]
                # T= T_dum
                # print(T)
                T = T*1000.
                # Tc = -np.dot(R.T,Tw)
                # print(K)
                # print(R)
                # print(R.T)
                our_cam['R'] = R
                print(our_cam['R'])
                our_cam['T'] = -np.dot(R.T,T)#T[[0,2,1]]   # the order to handle rotation and translation is reversed
                print(our_cam['T'])
                our_cam['fx'] = K[0]
                our_cam['fy'] = K[1]
                our_cam['cx'] = K[2]
                our_cam['cy'] = K[3]
                our_cam['k'] = np.array([0,0,0]).reshape(3,1)#np.zeros((3,)).reshape(3, 1)
                our_cam['p'] = np.array([0,0]).reshape(2,1)
                cameras[seq][k] = our_cam
        return cameras
    def project_3d(self,cam,poses_3d):
        poses_2d = np.zeros((len(poses_3d),17,3))
        for i,v in enumerate(poses_3d):

            points_2d = project_pose(v[:,:3],cam)
            points_2d_coco = self.points_to_coco17_cpu(points_2d)
            poses_2d[i] = points_2d_coco
        return poses_2d
    def points_to_coco17(self,points2d):
        new_points = points2d.cpu().numpy()
        new_points = np.delete(new_points,obj=(0,2),axis=0)
        coco_pose = np.zeros((17,3))
        coco_pose[0,:2] = new_points[0] 
        coco_pose[0,2] =1
        coco_pose[5:,2] = 1
        coco_pose[5,:2] = new_points[1]
        coco_pose[7,:2] = new_points[2]
        coco_pose[9,:2] = new_points[3]
        coco_pose[11,:2] = new_points[4]
        coco_pose[13,:2] = new_points[5]
        coco_pose[15,:2] = new_points[6]
        coco_pose[6,:2] = new_points[7]
        coco_pose[8,:2] = new_points[8]
        coco_pose[10,:2] = new_points[9]
        coco_pose[12,:2] = new_points[10]
        coco_pose[14,:2] = new_points[11]
        coco_pose[16,:2] = new_points[12]
        return coco_pose
    def points_to_coco17_cpu(self,points2d):
        new_points = points2d.numpy()
        new_points = np.delete(new_points,obj=(0,2),axis=0)
        coco_pose = np.zeros((17,3))
        coco_pose[0,:2] = new_points[0] 
        coco_pose[0,2] =1
        coco_pose[5:,2] = 1
        coco_pose[5,:2] = new_points[1]
        coco_pose[7,:2] = new_points[2]
        coco_pose[9,:2] = new_points[3]
        coco_pose[11,:2] = new_points[4]
        coco_pose[13,:2] = new_points[5]
        coco_pose[15,:2] = new_points[6]
        coco_pose[6,:2] = new_points[7]
        coco_pose[8,:2] = new_points[8]
        coco_pose[10,:2] = new_points[9]
        coco_pose[12,:2] = new_points[10]
        coco_pose[14,:2] = new_points[11]
        coco_pose[16,:2] = new_points[12]
        return coco_pose
    def points_to_coco17_3d_save_pos(self,points3d):
        # new_points = points3d.cpu().numpy()
        points_to_save = np.zeros((2,3))
        for i,v in enumerate(points3d):
            # points_to_save = np.zeros((2,3))
            points_to_save[0] = v[0,:3].cpu().numpy()
            points_to_save[1] = v[2,:3].cpu().numpy()
        return points_to_save
    def points_3d_to_coco_cpu(self,points3d):
        points_coco = np.zeros((points3d.shape[0],17,3))
        points_3d_numpy = points3d.cpu().numpy()
        points_3d_numpy = np.delete(points_3d_numpy,obj=(0,2),axis=1)
        bbox_3d =np.zeros((points3d.shape[0],5))

        for i,v in enumerate(points_3d_numpy):
            # print(v.shape)
            v2 = v[:,:3].copy()
            points_coco[i,0,:] = v2[0] 
            bbox_3d[i,0] = v2[:,0].min()
            bbox_3d[i,1] = v2[:,1].min()
            bbox_3d[i,2] = v2[:,0].max()
            bbox_3d[i,3] = v2[:,1].max()
            bbox_3d[i,4] = 1.#v[:,4][0].copy()*10
            # points_coco[0,2] =1
            # points_coco[5:,2] = 1
            points_coco[i,5,:] = v2[1]
            points_coco[i,7,:] = v2[2]
            points_coco[i,9,:] = v2[3]
            points_coco[i,11,:] = v2[4]
            points_coco[i,13,:] = v2[5]
            points_coco[i,15,:] = v2[6]
            points_coco[i,6,:] = v2[7]
            points_coco[i,8,:] = v2[8]
            points_coco[i,10,:] = v2[9]
            points_coco[i,12,:] = v2[10]
            points_coco[i,14,:] = v2[11]
            points_coco[i,16,:] = v2[12]
        return points_coco[:,:,[0,2,1]]/1000.,bbox_3d
    def post_proc_3d(self,points3d,bbox_3d):
        to_del = []
        for i,v in enumerate(points3d):
            # print(v)
            j=v[~np.all(v == 0, axis=1)]
            if not j.any():
                to_del.append(i)
        # print(to_del)
        if to_del:
            # print(to_del)
            points3d = np.delete(points3d,obj=tuple(to_del),axis=0)
            bbox_3d = np.delete(bbox_3d,obj=tuple(to_del),axis=0)

        return points3d,bbox_3d
        # coco_pose[15,:2] = new_points[4]
    @staticmethod
    def xyxy(x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[0] = int(x[0])   # x top
        y[1] = int(x[1])   # y top
        y[2] = int(x[2] + x[0])  # width
        y[3] = int(x[3] + x[1])  # height
        return y
    def generate_input_heatmap(self, joints, joints_vis=None):
        num_joints = joints[0].shape[0]
        target = np.zeros((num_joints, self.heatmap_size[1],\
                           self.heatmap_size[0]), dtype=np.float32)
        feat_stride = self.image_size / self.heatmap_size

        for n in range(len(joints)):
            human_scale = 2 * self.compute_human_scale(
                    joints[n][:, :2] / feat_stride, np.ones(num_joints))
            if human_scale == 0:
                continue

            cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
            tmp_size = cur_sigma * 3
            for joint_id in range(num_joints):
                if joints_vis is not None and joints_vis[n][joint_id] == 0:
                    continue

                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1]\
                        or br[0] < 0 or br[1] < 0:
                    continue

                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2

                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                # data augmentation
                if self.data_augmentation:
                    # random scaling
                    scale = 0.9 + np.random.randn(1) * 0.03 if random.random() < 0.6 else 1.0
                    if joint_id in [7, 8]:
                        scale = scale * 0.5 if random.random() < 0.1 else scale
                    elif joint_id in [9, 10]:
                        scale = scale * 0.2 if random.random() < 0.1 else scale
                    else:
                        scale = scale * 0.5 if random.random() < 0.05 else scale
                    g *= scale

                    # random occlusion
                    start = [int(np.random.uniform(0, self.heatmap_size[1] -1)),
                                int(np.random.uniform(0, self.heatmap_size[0] -1))]
                    end = [int(min(start[0] + np.random.uniform(self.heatmap_size[1] / 4, 
                            self.heatmap_size[1] * 0.75), self.heatmap_size[1])),
                            int(min(start[1] + np.random.uniform(self.heatmap_size[0] / 4,
                            self.heatmap_size[0] * 0.75), self.heatmap_size[0]))]
                    g[start[0]:end[0], start[1]:end[1]] = 0.0

                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                target[joint_id][img_y[0]:img_y[1],
                                    img_x[0]:img_x[1]] = np.maximum(
                                        target[joint_id][img_y[0]:img_y[1],
                                                        img_x[0]:img_x[1]],
                                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
            target = np.clip(target, 0, 1)

        return target


    @staticmethod
    def associate_bboxes(bbox,bboxes_2d):
        imin=-1
        # for ii, box in enumerate(bboxes_2d):

        mind = 0
        #     imin = -1
        for jj, box in enumerate(bboxes_2d):
            box_cor = [box[0],box[1],box[2]-box[0],box[3]-box[1]]
            iou = VoxelMethod.bb_intersection_over_union(bbox, box_cor)
            # print(iou)
            if iou>0. and iou<1.:
                if iou > mind:
                    mind = iou
                    imin = jj
            # polys_tr.append(polys[imin])
            # masks_tr.append(masks_cropped[imin])
        #print(mind)
        if imin!=-1:
            return imin,mind
        else:
            return None,None
            # tids.append(tid)

    @staticmethod
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

    @staticmethod
    def reorder_poses(pose1,pose2):
        cor_order = []
        for ind,pose in enumerate(pose1):
            for ind_,pose_ in enumerate(pose2):
                # print(pose)
                # print(pose_)
                if pose.all()==pose_.all():
                    if ind_ not in cor_order:
                        cor_order.append(ind_)
        return cor_order

if __name__ == "__main__":
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'testing')
    
    model = mdls.voxelpose.get(config,is_train=False)
    model.load_state
    # print(model)
