'''
written by Hsu, Mao-Huan 
started at 2021/6/8
'''
import os.path as osp
import numpy as np

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from core.yolov4_tools.model import yolov4
from core.yolov4_tools.yolo_utils import detect
from core.yolov4_tools.utils import get_anchors, get_classes

class ObjectDetector():
    def __init__(self, model_name, input_shape):
        anchors_file = osp.join(r'core\models', model_name, 'anchors.txt')
        classes_file = osp.join(r'core\models', model_name, 'classes.txt')
        weights_file = osp.join(r'core\models', model_name, 'weights.h5')

        self.model_name = model_name
        self.input_shape = input_shape
        self.anchors = get_anchors(anchors_file)
        self.classes = get_classes(classes_file)
        self.num_anchors = len(self.anchors)
        self.num_classes = len(self.classes)

        self.model = yolov4(Input(shape=self.input_shape+(3,)), self.num_anchors//3, self.num_classes)
        self.model.load_weights(weights_file)
        print(f'<{model_name}> model loaded.')

    def printModel(self):
        self.model.summary()

    def plotModel(self):
        return plot_model(self.model, f'{self.model_name}.png', True)

    def loadImage(self, image, crop=False, box=None):
        if crop:
            self.image = image[box[0, 1]:box[1, 1]+1, box[0, 0]:box[1, 0]+1] # [y1:y2+1, x1:x2+1] 
        else:
            self.image = image

        return self.image

    def runDetection(self, mode=None, bfilter=None, multi_vc=False):
        _, boxes, scores, classes = detect(self.model, self.image, self.input_shape, self.classes, self.anchors, 0.6, 0.4)

        if len(boxes) != 0:
            boxes, classes = self.restoreValues(boxes, classes)
            boxes, classes = self.obj_filter(boxes, scores, classes, mode, bfilter, multi_vc)
        else:
            boxes, classes = [], []

        return boxes, classes

    def restoreValues(self, boxes, classes):
        w, h = self.image.shape[1], self.image.shape[0]
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, 2, 2).astype(int)
        else:
            boxes = boxes.reshape(len(boxes), 2, 2).astype(int)
        for i, box in enumerate(boxes):
            for j, [x, y] in enumerate(box):
                if x < 0: x = 0
                elif x >= w: x = w-1
                if y < 0: y = 0
                elif y >= h: y = h-1
                boxes[i][j] = [x, y]

        classes = [self.classes[c] for c in classes]

        return boxes, classes

    def obj_filter(self, boxes, scores, classes, mode, bfilter, multi_vc):
        vehicle_types = ['car', 'bus', 'truck']
        max_area = 0
        max_score = 0.
        max_id = None
        delete_id = []

        if mode == 'vehicle':
            for i in range(len(classes)):
                if classes[i] not in vehicle_types:
                    delete_id.append(i)
            if len(delete_id) != 0:
                boxes = np.delete(boxes, delete_id, axis=0)
                scores = np.delete(scores, delete_id)
                classes = np.delete(classes, delete_id)

            if not multi_vc:
                if len(classes) == 1:
                    return boxes, classes[0]

                if bfilter == 'max_area': 
                    for i, b in enumerate(boxes):
                        area = (b[1, 0]-b[0, 0]) * (b[1, 1]-b[0, 1])
                        if area > max_area:
                            max_area = area
                            max_id = i
                elif bfilter == 'max_score':
                    for i in range(len(scores)):
                        if scores[i] > max_score:
                            max_score = scores[i]
                            max_id = i
                if max_id != None:
                    boxes = [boxes[max_id]]
                    classes = classes[max_id]

        elif mode == 'plate':
            if len(classes) == 1:
                return boxes[0], classes[0]

            for i in range(len(scores)):
                if scores[i] > max_score:
                    max_score = scores[i]
                    max_id = i

            if max_id != None:
                boxes = boxes[max_id]
                classes = classes[max_id]

        return boxes, classes
