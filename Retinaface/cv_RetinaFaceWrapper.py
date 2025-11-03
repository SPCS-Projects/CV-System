def detect_face(data, img, im_info, im_scale, align):

    from Retinaface.cv_retinaface import detect_faces
    import Retinaface.cv_imgprocess as process

    resp = []

    obj = detect_faces(data, img, im_info, im_scale)

    if type(obj) == dict:
        for key in obj:
            identity = obj[key]
            facial_area = identity["facial_area"]

            y = facial_area[1]
            h = facial_area[3] - y
            x = facial_area[0]
            w = facial_area[2] - x
            img_region = [x, y, w, h]

            #detected_face = img[int(y):int(y+h), int(x):int(x+w)] #opencv
            detected_face = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

            if align:
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                #mouth_right = landmarks["mouth_right"]
                #mouth_left = landmarks["mouth_left"]

                detected_face = process.alignment_procedure(detected_face, right_eye, left_eye, nose)

            resp.append((detected_face, img_region))

    return resp
