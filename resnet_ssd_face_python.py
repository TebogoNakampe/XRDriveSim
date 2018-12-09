import pyrealsense2 as rs
import numpy as np
import argparse
import cv2
from cv2 import dnn

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)



inWidth = 300
inHeight = 300
confThreshold = 0.5

prototxt = 'deploy.prototxt'
caffemodel = 'res10_300x300_ssd_iter_140000.caffemodel'

if __name__ == '__main__':
    net = dnn.readNetFromCaffe(prototxt, caffemodel)
    #cap = cv.VideoCapture(0)
    while True:
        ret, frames = pipeline.wait_for_frames()
        #frame = frames.as_stream()
        #color_frame = frames.get_color_frame()
        # frame = cap.read()
        cols = 640
        rows = 480
        frame = np.asanyarray(frames.get_data())

        net.setInput(dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
        detections = net.forward()

        perf_stats = net.getPerfProfile()

        #print('Inference time, ms: %.2f' % (perf_stats[0] / cv.getTickFrequency() * 1000))

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                             (0, 255, 0))
                label = "face: %.4f" % confidence
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                    (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                    (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("detections", frame)
        if cv.waitKey(1) != -1:
            break
