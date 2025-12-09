import cv2

cap = cv2.VideoCapture("udpsrc port=5000 ! application/x-rtp,payload=96,encoding-name=H264 !\
    rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink drop=1", cv2.CAP_GSTREAMER);
while True:
    ret,frame= cap.read()
    print(ret)
