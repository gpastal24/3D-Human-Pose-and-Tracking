#!/bin/bash
# ldconfig
#cd 3d_cams/
ARGS=$(cat args)

python3 opencv_zed_stream.py ${ARGS} 


