# 3D Human Pose Estimation using ZED Cameras

This project implements a real-time 3D human pose estimation system using multiple ZED cameras. It captures video streams, detects human joints in 2D, and reconstructs them in 3D space. The system also includes functionalities for tracking individuals across multiple cameras and broadcasting the results through a RabbitMQ messaging system.

## Features

- **Real-time 3D Pose Estimation:** Captures video streams from multiple ZED cameras and performs 3D human pose estimation in real-time.
- **Multi-camera Support:** Can be configured to use multiple cameras for a more robust and accurate 3D reconstruction.
- **Human Tracking:** Implements a tracking system to maintain the identity of individuals across multiple cameras. Two tracking solutions are integrated in the system.
- **RabbitMQ Integration:** Broadcasts the 3D pose data through a RabbitMQ messaging system for easy integration with other applications.
- **TRT Optimization:** Supports TensorRT optimization for faster inference.

## Dependencies

The project has the following dependencies:

- Python 3.8+
- PyTorch
- OpenCV
- RabbitMQ
- TensorRT (optional)

To install the required Python packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the main script, you need to provide a configuration file as an argument. The configuration file specifies the camera parameters, the 3D space dimensions, and other settings.

```bash
python opencv_zed_stream.py --cfg FasterVP/FasterVoxelPose/configs/panoptic/test_strintzis.yaml
```

### Configuration

The main script accepts the following arguments:

- `--cfg`: Path to the configuration file.
- `--inside`: Boolean flag to indicate if the system is running inside or outside.
- `--cnc_host`: RabbitMQ host.
- `--rabbitcncuser`: RabbitMQ user.
- `--rabbitcncpass`: RabbitMQ password.
- `--cnc_port`: RabbitMQ port.
- `--exchange_stream`: RabbitMQ exchange stream.
- `--routing_key`: RabbitMQ routing key.
- `--enable_rabbitcnc`: Enable RabbitMQ integration.
- `--cams_n`: Number of cameras to use.
- `--usb`: Use USB cameras.
- `--video`: Use video files as input.
- `--stream`: Use a video stream as input.
- `--tracking_cam`: The camera to use for tracking (Siamese tracker).
- `--use_bytetrack`: Use ByteTrack for tracking on Bird's Eye View.
- `--log_freq`: Logging frequency.
- `--visualize_debug`: Enable debug visualization.
- `--cam_list`: List of camera serial numbers to use.
- `--trt`: Enable TensorRT optimization.
- `--test_mode`: Run in test mode.
- `--device`: The device to use for inference (e.g., `cuda:0` or `cpu`).
- `--save_video`: Save the output videos.

## Scripts

### `run.sh`

This script is used to run the main application. It reads the arguments from the `args` file and passes them to the `opencv_zed_stream.py` script.

To use it, first, populate the `args` file with the desired arguments, and then run the script:

```bash
bash run.sh
```

### `generate_trt_weights.sh`

This script is used to generate the TensorRT weights for the model. It reads the arguments from the `args` file and passes them to the `trt_conv_FVP.py` script.

To use it, first, populate the `args` file with the desired arguments, and then run the script:

```bash
bash generate_trt_weights.sh
```
