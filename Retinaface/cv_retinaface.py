import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import Retinaface.cv_imgprocess as process
import cv2
import tensorflow as tf



def detect_faces(data, img, im_info, im_scale, threshold=0.9):
    nms_threshold = 0.4; decay4=0.5

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        'stride32': np.array([[-248., -248.,  263.,  263.], [-120., -120.,  135.,  135.]], dtype=np.float32),
        'stride16': np.array([[-56., -56.,  71.,  71.], [-24., -24.,  39.,  39.]], dtype=np.float32),
        'stride8': np.array([[-8., -8., 23., 23.], [ 0.,  0., 15., 15.]], dtype=np.float32)
    }

    _num_anchors = {'stride32': 2, 'stride16': 2, 'stride8': 2}

    #---------------------------

    proposals_list = []
    scores_list = []
    landmarks_list = []

    tf_array = []
    for value in data:
        changed_mb = np.moveaxis(value, 1, -1)
        temp = tf.convert_to_tensor(changed_mb)
        tf_array.append(temp)
    net_out = [elt.numpy() for elt in tf_array]

    sym_idx = 0
    for _idx, s in enumerate(_feat_stride_fpn):
        _key = 'stride%s'%s
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors['stride%s'%s]:]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

        A = _num_anchors['stride%s'%s]
        K = height * width
        anchors_fpn = _anchors_fpn['stride%s'%s]
        anchors = process.anchors_plane(height, width, s, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.reshape((-1, 1))

        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_deltas = bbox_deltas
        bbox_pred_len = bbox_deltas.shape[3]//A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:,0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:,1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:,2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:,3::4] * bbox_stds[3]
        proposals = process.bbox_pred(anchors, bbox_deltas)

        proposals = process.clip_boxes(proposals, im_info[:2])
        if s==4 and decay4<1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel>=threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)
        # Commented out because landmark is just for face alignment
        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3]//A
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len//5))
        landmarks = process.landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)
    # Commented out because landmark is just for face alignment
    if proposals.shape[0]==0:
        landmarks = np.zeros( (0,5,2) )
        return np.zeros( (0,5) ), landmarks
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    # Commented out because landmark is just for face alignment
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)

    #nms = cpu_nms_wrapper(nms_threshold)
    #keep = nms(pre_det)
    keep = process.cpu_nms(pre_det, nms_threshold)

    det = np.hstack( (pre_det, proposals[:,4:]) )
    det = det[keep, :]
    # Commented out because landmark is just for face alignment
    landmarks = landmarks[keep]

    resp = {}
    for idx, face in enumerate(det):

        label = 'face_'+str(idx+1)
        resp[label] = {}
        resp[label]["score"] = face[4]

        resp[label]["facial_area"] = list(face[0:4].astype(int))

        # Commented out because landmark is just for face alignment
        resp[label]["landmarks"] = {}
        resp[label]["landmarks"]["right_eye"] = list(landmarks[idx][0])
        resp[label]["landmarks"]["left_eye"] = list(landmarks[idx][1])
        resp[label]["landmarks"]["nose"] = list(landmarks[idx][2])
        resp[label]["landmarks"]["mouth_right"] = list(landmarks[idx][3])
        resp[label]["landmarks"]["mouth_left"] = list(landmarks[idx][4])
    return resp


def detect(data, img, im_info, im_scale, align):
    """
    Direct wrapper around detect_faces() that returns a list of (detected_face, img_region).
    Replaces dependency on cv_RetinaFaceWrapper.detect_face.
    """
    resp = []
    obj = detect_faces(data, img, im_info, im_scale)
    if type(obj) == dict and len(obj) > 0:
        for key in obj:
            identity = obj[key]
            facial_area = identity.get("facial_area", None)
            if facial_area is None or len(facial_area) < 4:
                continue
            x1, y1, x2, y2 = map(int, facial_area[:4])
            w = x2 - x1
            h = y2 - y1
            img_region = [x1, y1, w, h]
            detected_face = img[y1:y2, x1:x2]

            if align and "landmarks" in identity:
                lm = identity["landmarks"]
                right_eye = lm.get("right_eye")
                left_eye = lm.get("left_eye")
                nose = lm.get("nose")
                if right_eye is not None and left_eye is not None and nose is not None:
                    detected_face = process.alignment_procedure(detected_face, right_eye, left_eye, nose)

            resp.append((detected_face, img_region))
    return resp


def test(testing_output, image, im_info, im_scale, align):
    detected_face, img_region = detect(testing_output, image, im_info, im_scale, align)
    if (isinstance(detected_face, np.ndarray)) or detected_face != None:
        return detected_face, img_region
    else:
        if detected_face == None:
            return None, ":)"
            # raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")


def resize(img, target_size):
    img_pixel = []

    for i in range(len(img)):
        factor_0 = target_size[0] / img[i].shape[0]
        factor_1 = target_size[1] / img[i].shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img[i].shape[1] * factor), int(img[i].shape[0] * factor))
        img[i] = cv2.resize(img[i], dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img[i].shape[0]
        diff_1 = target_size[1] - img[i].shape[1]

        img[i] = np.pad(img[i], ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                        'constant')

        # double check: if target image is not still the same size with target.
        if img[i].shape[0:2] != target_size:
            img[i] = cv2.resize(img[i], target_size)

        # ---------------------------------------------------

        # normalizing the image pixels

        img_pixels = np.asarray(img[i], dtype=np.float32)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # normalize input in [0, 1]
        img_pixel.append(img_pixels)
        # ---------------------------------------------------
    return img_pixel


def get_data(testing_output, img, im_info, im_scale, align, target_size=(160, 160)):
    img_pixel = []
    img, region = test(testing_output, img, im_info, im_scale, align)
    if img is None:
        return None
    age_gender = resize(img, (224, 224))
    face_det = resize(img, target_size)
    return face_det, age_gender
