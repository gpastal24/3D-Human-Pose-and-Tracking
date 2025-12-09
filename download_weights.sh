#! /bin/bash
mkdir weights
gdown https://drive.google.com/drive/folders/1F1tEUcHHgL_CxuOI6N3qubCRCaLYrqHv?usp=drive_link -O weights --folder
mkdir FasterVP/FasterVoxelPose/output/panoptic/voxelpose_50/jln64/panoptic_5_cams_allseq/
mkdir FasterVP/FasterVoxelPose/models/
mv weights/pose_resnet50_panoptic.pth.tar FasterVP/FasterVoxelPose/models/
mv weights/model_best.pth.tar FasterVP/FasterVoxelPose/output/panoptic/voxelpose_50/jln64/panoptic_5_cams_allseq/
mv weights/SiamRPNOTB.model test_SiamRPN/
rm -r weights