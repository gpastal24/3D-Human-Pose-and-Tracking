#!/bin/bash
ldconfig
ARGS=$(cat args)
python3 trt_conv_FVP.py ${ARGS} 



