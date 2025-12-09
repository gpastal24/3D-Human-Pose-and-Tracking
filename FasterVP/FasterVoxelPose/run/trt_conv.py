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
import _init_paths
from core.config import config, update_config
from utils.utils import create_logger
from utils.vis import save_debug_2d_images
import dataset
import models

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
    cfg = args.cfg
    update_config(cfg)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformss = transforms.Compose([
        transforms.ToTensor(),
        normalize,])
    # logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'trt_test')

    # model = get_voxelpose(config)
    model =eval('models.' + config.MODEL + '.get')(config, is_train=False)
    # net= get(config)
    # model = eval('models.' + config.MODEL + '.get')(config, is_train=False)
    print(model)
    device = 'cuda:0'
    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.load_state_dict(torch.load(test_model_file, map_location=torch.device(device)))
    else:
        raise ValueError('Check the model file for testing!')
    model.to(device=device).eval()
    backbone = model.backbone
    # backbone=backbone.to(device='cuda:1').eval()
    # print(backbone.device)
    x = torch.ones((1,3,512,960)).to(device=device)
    backbone_trt = torch2trt(backbone,[x],fp16_mode=True,max_batch_size=5)
    torch.save(backbone_trt.state_dict(),"trt_backbone.pth")
    print('Saved trt backbone!')
    # model.backbone = TRTModule()
    # model.backbone.load_state_dict(torch.load("/home/gpastal/3d_pose_voxels/Faster-VoxelPose/trt_backbone.pth"))
    # model.eval()