import numpy as np
import cv2
from PIL import Image
import math

#this function is copied from the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
def resize_image(img, scales):
    img_w, img_h = img.shape[0:2]
    target_size = scales[0]
    max_size = scales[1]

    if img_w > img_h:
        im_size_min, im_size_max = img_h, img_w
    else:
        im_size_min, im_size_max = img_w, img_h

    im_scale = target_size / float(im_size_min)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = max_size / float(im_size_max)

    if im_scale != 1.0:
        img = cv2.resize(
            img,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )
    return img, im_scale

#this function is copied from the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
def preprocess_image(img):
    pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pixel_scale = float(1.0)
    scales = [img.shape[0], img.shape[1]]
    img, im_scale = resize_image(img, scales)
    img = img.astype(np.float32)
    im_tensor = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

    # Make image scaling + BGR2RGB conversion + transpose (N,H,W,C) to (N,C,H,W)
    for i in range(3):
        im_tensor[0, :, :, i] = (img[:, :, 2 - i] / pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]
    return im_tensor, img.shape[0:2], im_scale

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

#this function copied from the deepface repository: https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
def alignment_procedure(img, left_eye, right_eye, nose):

    #this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    #-----------------------
    upside_down = False
    if nose[1] < left_eye[1] or nose[1] < right_eye[1]:
        upside_down = True

    #-----------------------
    #find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock

    #-----------------------
    #find length of triangle edges

    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    #-----------------------

    #apply cosine rule

    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree

        #-----------------------
        #rotate base image

        if direction == -1:
            angle = 90 - angle

        if upside_down == True:
            angle = angle + 90

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    #-----------------------

    return img #return img anyway

#this function is copied from the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
def bbox_pred(boxes, box_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1]>4:
        pred_boxes[:,4:] = box_deltas[:,4:]

    return pred_boxes

# This function copied from the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
def landmark_pred(boxes, landmark_deltas):
    if boxes.shape[0] == 0:
      return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:,i,0] = landmark_deltas[:,i,0]*widths + ctr_x
        pred[:,i,1] = landmark_deltas[:,i,1]*heights + ctr_y
    return pred

# This function copied from rcnn module of retinaface-tf2 project: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/processing/bbox_transform.py
def clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

#this function is mainly based on the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/cython/anchors.pyx
def anchors_plane(height, width, stride, base_anchors):
    A = base_anchors.shape[0]
    c_0_2 = np.tile(np.arange(0, width)[np.newaxis, :, np.newaxis, np.newaxis], (height, 1, A, 1))
    c_1_3 = np.tile(np.arange(0, height)[:, np.newaxis, np.newaxis, np.newaxis], (1, width, A, 1))
    all_anchors = np.concatenate([c_0_2, c_1_3, c_0_2, c_1_3], axis=-1) * stride + np.tile(base_anchors[np.newaxis, np.newaxis, :, :], (height, width, 1, 1))
    return all_anchors

#this function is mainly based on the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/cython/cpu_nms.pyx
#Fast R-CNN by Ross Girshick
def cpu_nms(dets, threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]; iy1 = y1[i]; ix2 = x2[i]; iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j]); yy1 = max(iy1, y1[j]); xx2 = min(ix2, x2[j]); yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1); h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= threshold:
                suppressed[j] = 1

    return keep
