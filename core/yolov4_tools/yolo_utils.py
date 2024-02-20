"""YOLO utility functions."""
import numpy as np
from core.yolov4_tools.utils import *

def detect(model, image, input_shape, all_classes, anchors, conf_th, nms_th):
    pimage = pre_process(image, input_shape)

    outs = predict(model, pimage, input_shape, len(all_classes))
    boxes, scores, classes = yolo_out(outs, input_shape, anchors, image.shape, conf_th, nms_th)

    return image, boxes, scores, classes

def predict(model, image, input_shape, num_classes):
    outs = model.predict(image)

    a1 = np.reshape(outs[0], (1, input_shape[0]//32, input_shape[1]//32, 3, 5+num_classes))
    a2 = np.reshape(outs[1], (1, input_shape[0]//16, input_shape[1]//16, 3, 5+num_classes))
    a3 = np.reshape(outs[2], (1, input_shape[0]//8, input_shape[1]//8, 3, 5+num_classes))

    return [a1, a2, a3]

def yolo_out(outs, input_shape, anchors, shape, t1, t2):
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    boxes, classes, scores = [], [], []

    for out, mask in zip(outs, masks):
        b, c, s = process_feats(out, input_shape, mask, anchors)
        b, c, s = boxes_filter(b, c, s, t1)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # Scale boxes back to original image shape.
    w, h = shape[1], shape[0]
    image_dims = [w, h, w, h]
    boxes = boxes * image_dims

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s, t2)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return [], [], []

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]

    return boxes, scores, classes
