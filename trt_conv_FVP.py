# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
import argparse
import os
from tqdm import tqdm
import cv2
# import _init_paths
from FasterVP.FasterVoxelPose.lib.core.config import config, update_config
from FasterVP.FasterVoxelPose.lib.utils.utils import create_logger
# from utils.vis import save_debug_2d_images
from FasterVP.FasterVoxelPose.lib.models.voxelpose import get_voxelpose
from FasterVP.voxel_pose_class import VoxelMethod

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, _ = parser.parse_known_args()
    update_config(args.cfg)

    return args
# from lib.models.voxelpose import VoxelPoseNet,get_voxelpose

if __name__ == "__main__":
    # main()
    from torch2trt import TRTModule, torch2trt
    args = parse_args()
    update_config(args.cfg)

    # logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'trt_test')
    with torch.no_grad():
        VP=VoxelMethod(config,transform=None)

        VP.load_weights("FasterVP/FasterVoxelPose/output/panoptic/voxelpose_50/jln64/panoptic_5_cams_allseq/model_best.pth.tar")
        VP.model.cuda().eval()
        backbone = VP.model.backbone
        P2PNet = VP.model.joint_net.conv_net
        HDNNet = VP.model.pose_net.center_net
        backbone.cuda().eval()
        P2PNet.cuda().eval()
        HDNNet.cuda().eval()
        device='cuda'
        x = torch.ones((1,3,512,960)).to(device=device)
        backbone_trt = torch2trt(backbone,[x],fp16_mode=True,use_onnx=True,min_shapes=[(1,3,512,960)],max_shapes=[(5,3,512,960)])
        torch.save(backbone_trt.state_dict(),"trt_backbone.pth")
        print('Saved trt backbone!')
        x_p = torch.ones((30, 15, 64, 64)).to(device=device)
        x_d = torch.ones((8, 15, 60, 60, 20)).to(device=device)
        p2p_trt = torch2trt(P2PNet,[x_p],fp16_mode=True,use_onnx=True,min_shapes=[(3,15,64,64)],max_shapes=[(30,15,64,64)])
        torch.save(p2p_trt.state_dict(),"trt_p2p.pth")
        print('Saved trt p2p!')

        center_trt = torch2trt(HDNNet,[x_d],fp16_mode=True,use_onnx=True,min_shapes=[(1,15,60,60,20)],max_shapes=[(10,15,60,60,20)])
        torch.save(center_trt.state_dict(),"trt_center_net.pth")
        print('Saved trt centernet!')

    # model.backbone = TRTModule()
    # model.backbone.load_state_dict(torch.load("/home/gpastal/3d_pose_voxels/Faster-VoxelPose/trt_backbone.pth"))
    # model.eval()
