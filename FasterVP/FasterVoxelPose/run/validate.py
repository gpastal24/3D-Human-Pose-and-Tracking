# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
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


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'validate')
    cfg_name = os.path.basename(args.cfg).split('.')[0]
    writer = SummaryWriter(log_dir=tb_log_dir)
    device = 'cuda:0'
    gpus = [int(i) for i in config.GPUS.split(',')]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    print('=> Constructing models...')
    model = eval('models.' + config.MODEL + '.get')(config, is_train=False)
    # print(model.backbone)
    # print(model.joint_net)
    # print(model.pose_net)
    with torch.no_grad():
        # model = torch.nn.DataParallel(model.cuda(), device_ids=gpus)
        # model.to(f'cuda:{model.device_ids[0]}')
        model.to(device)


    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        # model.module.load_state_dict(torch.load(test_model_file, map_location=torch.device(device)))
        model.load_state_dict(torch.load(test_model_file, map_location=torch.device(device)))

    else:
        raise ValueError('Check the model file for testing!')

    print("=> Validating...")
    from torch2trt import TRTModule
    # model.eval()
    # model.backbone = TRTModule()
    # model.backbone.load_state_dict(torch.load("/home/gpastal/3d_pose_voxels/Faster-VoxelPose/trt_backbone.pth"))
    # model.backbone.eval()
    # model.module.backbone.cuda(1)
    model.eval()
    # print(model.device_ids)
    # loading constants of the dataset
    cameras = test_loader.dataset.cameras
    # resize_transform = torch.as_tensor(test_loader.dataset.resize_transform, dtype=torch.float, device='cuda:{}'.format(model.device_ids[0]))
    resize_transform = torch.as_tensor(test_loader.dataset.resize_transform, dtype=torch.float, device=device)
    # print(resize_transform)
    timing = 0
    with torch.no_grad():
        all_final_poses = []
        starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)

        for i, (inputs, _, meta, input_heatmap) in enumerate(tqdm(test_loader)):
            starter.record()
            # print(cameras)
            # print(input_heatmap)
            # inputs.cuda(1)
            if config.DATASET.TRAIN_HEATMAP_SRC == 'image':
                final_poses, poses, proposal_centers, _, input_heatmap = model(views=inputs.to(device), meta=meta, cameras=cameras, resize_transform=resize_transform)
            else:
                final_poses, poses, proposal_centers, _, _ = model(meta=meta, input_heatmaps=input_heatmap, cameras=cameras, resize_transform=resize_transform)
            # print(final_poses)
            # print(cameras)
            final_poses = final_poses.cpu().numpy()
            # print(input_heatmap.size())
            # print(final_poses)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            # print(curr_time)
            timing+=curr_time

            # heat= input_heatmap.cpu()
            # heat=heat.transpose(1,4).transpose(1,2)
            # heat=heat.numpy()
            # print(heat.shape)
            # for ran in heat:
            #     # temp = heat[ran]
            #     # print(ran.shape)
            #     for ran2 in ran:
            #         # temp2 = temp[ran2]
            #         # cv2.imshow('',ran2)
            #         cv2.waitKey(100)
            #         cv2.imwrite('test.png',ran2)
            #         # time.sleep(0.1)
            for b in range(final_poses.shape[0]):
                all_final_poses.append(final_poses[b])

            prefix = '{}_{:08}'.format(os.path.join(final_output_dir, 'validation'), i)
            # print(meta)
            # save_debug_2d_images(config, meta, final_poses, poses.cpu().numpy(), proposal_centers.cpu().numpy(), prefix)
        print(timing/i)
    if test_dataset.has_evaluate_function:
        metric, msg = test_loader.dataset.evaluate(all_final_poses)
        logger.info(msg)

if __name__ == "__main__":
    main()
