import os
import argparse
import time
import cv2
from utils import *
from darknet import Darknet

parser = argparse.ArgumentParser()
parser.add_argument("camera_index", help="Index number of camera (web-cam is usually 0)")
parser.add_argument("save_path", help="Path where you want to save the video")
parser.add_argument("duration", help="recording duration in seconds")
parser.add_argument("fps", help="frames per second")
args = parser.parse_args()

INDEX = args.camera_index
PATH = args.save_path
DURATION = args.duration
FPS = args.fps
DELAY = (1/FPS) * 0.95 #delay in seconds

class Streamer():
    """
    Recording daemon
    """
    def __init__(self, ind, duration, fps, save_path, delay):
        self.index = ind
        self.duration = duration
        self.fps = fps
        self.save_path = save_path
        self.delay = delay

    def start(self):
        """
        Records frame by frame
        """
        frames = []
        cap = cv2.VideoCapture(self.index)
        start_time = time.time()

        while time.time()-start_time < self.duration:
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            time.sleep(self.delay)

        cap.release()
        return frames

    def capture(self):
        """
        Start capturing
        """
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(os.path.join(self.save_path, 'output.avi'), fourcc, 20.0, (640, 480))

        frames = self.start()
        cfg_file = './cfg/yolov3.cfg'
        weight_file = './weights/yolov3.weights'
        namesfile = 'data/coco.names'

        m = Darknet(cfg_file)
        m.load_weights(weight_file)

        class_names = load_class_names(namesfile)
        nms_thresh = 0.6
        iou_thresh = 0.4

        for frame in frames:
            resized_frame = cv2.resize(frame, (m.width, m.height))
            boxes = detect_objects(m, resized_frame, iou_thresh, nms_thresh)
            frame = plot_boxes(frame, boxes, class_names, plot_labels=True)

            frame = cv2.flip(frame, 0)
            out.write(frame)

        out.release()


S = Streamer(INDEX, DURATION, FPS, PATH, DELAY)
S.capture()
