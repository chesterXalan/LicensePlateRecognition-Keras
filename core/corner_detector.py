'''
written by Hsu, Mao-Huan 
started at 2022/4/15
'''
import os.path as osp
import cv2
import numpy as np

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

class CornerDetector():
    def __init__(self, model_name):
        model_file = osp.join(r'core\models', model_name, 'model.h5')
        self.model = load_model(model_file, compile=False)
        print(f'<{model_name}> model loaded.')

        self.model_name = model_name
        self.w, self.h = self.model.input_shape[1:3][::-1] 
        self.ow, self.oh = (100, 50)

    def printModel(self):
        self.model.summary()

    def plotModel(self):
        return plot_model(self.model, f'{self.model_name}.png', True)

    def loadImage(self, image, crop=False, pts=None):
        if crop:
            pts = pts.copy().reshape(2, 2)
            offset_x, offset_y = int(0.2*(pts[1, 0]-pts[0, 0])), int(0.2*(pts[1, 1]-pts[0, 1]))
            pts[0, 0], pts[0, 1] = pts[0, 0]-offset_x, pts[0, 1]-offset_y
            pts[1, 0], pts[1, 1] = pts[1, 0]+offset_x, pts[1, 1]+offset_y

            for i, [x, y] in enumerate(pts):
                if x < 0: x = 0
                elif x >= image.shape[1]: x = image.shape[1]-1
                if y < 0: y = 0
                elif y >= image.shape[0]: y = image.shape[0]-1
                pts[i] = [x, y]

            image = image[pts[0, 1]:pts[1, 1]+1, pts[0, 0]:pts[1, 0]+1]
        self.image = cv2.resize(image, (self.w, self.h))

        return image

    def runDetection(self):
        image = np.array([self.image]).astype(np.float32)/255
        pred = self.model.predict(image)[0]

        pts = self.restoreValues(pred.copy())
        res = self.perspective(self.image, pts)

        return res.astype('uint8'), pts.astype('int')

    def restoreValues(self, pts):
        pts = pts.reshape(4, 2)
        pts[:, 0] = np.round(pts[:, 0]*self.w)
        pts[:, 1] = np.round(pts[:, 1]*self.h)

        for i in range(4):
            if pts[:, 0][i] < 0: pts[:, 0][i] = 0
            elif pts[:, 0][i] >= self.w: pts[:, 0][i] = self.w-1
            if pts[:, 1][i] < 0: pts[:, 1][i] = 0
            elif pts[:, 1][i] >= self.h: pts[:, 1][i] = self.h-1

        return pts

    def perspective(self, image, pts):
        res = np.array([[0, 0], [self.ow, 0], [self.ow, self.oh], [0, self.oh]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(pts, res)
        res = cv2.warpPerspective(image, m, (self.ow, self.oh))

        return res
