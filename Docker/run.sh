QT_DEBUG_PLUGINS=1 docker run \
    -e QT_XCB_GL_INTEGRATION=xcb_egl \
    -e DISPLAY=:1 \
    --user $(id -u):$(id -g) \
    --userns=host \
    --net=host \
    --ipc=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${HOME}/.Xauthority:/home/$(whoami)/.Xauthority:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    --gpus all \
    --privileged \
    -it test_gtk:latest
~

