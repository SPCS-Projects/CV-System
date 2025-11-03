import cv2
import depthai as dai
import numpy as np
import Retinaface.cv_imgprocess as process
from Retinaface.cv_retinaface import get_data
from ntplib import NTPClient
import time
from datetime import datetime
from cv_db import sql
# from cv_attributes import statistics
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import os


def createDirs(path):
    if not os.path.isfile(path + "/log.txt"):
        f = open(path + "/log.txt", "x")
        f.close()
    if not os.path.exists(path + "/database"):
        os.mkdir(path + "/database")
    if not os.path.exists(path + "/images"):
        os.mkdir(path + "/images")
    if not os.path.exists(path + "/faces"):
        os.mkdir(path + "/faces")
    if not os.path.exists(path + "/db_backup"):
        os.mkdir(path + "/db_backup")
    return True


def getPath(__file__):
    return os.path.dirname(os.path.abspath(__file__))


def getTime(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%d:%02d:%02d" % (hour, min, sec)


def cv_main(que, path, flip):
    createDirs(path)
    pipeline = dai.Pipeline()
    # Define a source - color camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setVideoSize(1280, 720)

    # cam
    xout_frame = pipeline.createXLinkOut()
    xout_frame.setStreamName("det_frame")
    cam_rgb.video.link(xout_frame.input)

    # -----------------------------------------------------------------#

    # tic = time.time()
    detection_nn = pipeline.createNeuralNetwork()
    detection_nn.setBlobPath(path + "/blobs/Retinaface-720x1280.blob")
    detection_nn.setNumInferenceThreads(1)

    xin_det = pipeline.createXLinkIn()
    xin_det.setStreamName("det_in")
    xin_det.out.link(detection_nn.input)

    xin_det.setMaxDataSize(3000000)
    xin_det.setNumFrames(1)
    xout_det = pipeline.createXLinkOut()
    xout_det.setStreamName("det_out")
    detection_nn.out.link(xout_det.input)

    # -----------------------------------------------------------------#

    landmark_nn = pipeline.createNeuralNetwork()
    landmark_nn.setBlobPath(path + "/blobs/Facenet.blob")

    xin_land = pipeline.createXLinkIn()
    xin_land.setStreamName("land_in")
    xin_land.out.link(landmark_nn.input)

    xin_land.setMaxDataSize(100000)
    xin_land.setNumFrames(1)
    xout_land = pipeline.createXLinkOut()
    xout_land.setStreamName("land_out")
    landmark_nn.out.link(xout_land.input)

    # -----------------------------------------------------------------#

    # age_nn = pipeline.createNeuralNetwork()
    # age_nn.setBlobPath(path + "/blobs/Race.blob")
    #
    # xin_age = pipeline.createXLinkIn()
    # xin_age.setStreamName("age_in")
    # xin_age.out.link(age_nn.input)
    #
    # xin_age.setMaxDataSize(200000)
    # xin_age.setNumFrames(1)
    # xout_age = pipeline.createXLinkOut()
    # xout_age.setStreamName("age_out")
    # age_nn.out.link(xout_age.input)

    # -----------------------------------------------------------------#
    date = datetime.now().strftime("%Y-%m-%d")
    database = path + '/database/{}.db'.format(date)

    with dai.Device(pipeline) as device:

        # device.setLogLevel(dai.LogLevel.DEBUG)
        # device.setLogOutputLevel(dai.LogLevel.INFO)
        q_frame = device.getOutputQueue(name="det_frame", maxSize=2, blocking=False)
        qIn = device.getInputQueue(name="det_in", maxSize=2, blocking=True)
        q_det = device.getOutputQueue(name="det_out", maxSize=2, blocking=True)
        landIn = device.getInputQueue(name="land_in", maxSize=2, blocking=True)
        landOut = device.getOutputQueue(name="land_out", maxSize=2, blocking=True)

        # ageIn = device.getInputQueue(name="age_in", maxSize=2, blocking=True)
        # ageOut = device.getOutputQueue(name="age_out", maxSize=2, blocking=True)
        time.sleep(20)
        que.put("start")
        started = time.time()
        while True:
            timestr = datetime.now().strftime("%H-%M-%S_%Y-%m-%d")
            target_img_path = timestr
            tic = time.time()
            try:
                time_stamp = datetime.fromtimestamp(
                    (NTPClient().request(host='time.kku.ac.th', version=3, port='ntp')).tx_time)
                # time_stamp = datetime.utcfromtimestamp((NTPClient().request(host='time.kku.ac.th', version=3, port='ntp')).tx_time)
            except:
                time_stamp = datetime.now()
            frame = q_frame.get().getCvFrame()
            if flip is True:
                image = cv2.flip(frame, 0)
            else:
                image = frame
                # cv2.imshow("CV System - preview", image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_processed, im_info, im_scale = process.preprocess_image(img_rgb)
            img_processed = np.moveaxis(img_processed, -1, 1)
            img = dai.ImgFrame()
            img.setData(img_processed)
            img.setHeight(image.shape[0])
            img.setWidth(image.shape[1])
            qIn.send(img)

            layers = ["StatefulPartitionedCall/model_1/tf.compat.v1.transpose_7/transpose",
                      "StatefulPartitionedCall/model_1/face_rpn_bbox_pred_stride32/BiasAdd/Add",
                      "StatefulPartitionedCall/model_1/face_rpn_landmark_pred_stride32/BiasAdd/Add",
                      "StatefulPartitionedCall/model_1/tf.compat.v1.transpose_9/transpose",
                      "StatefulPartitionedCall/model_1/face_rpn_bbox_pred_stride16/BiasAdd/Add",
                      "StatefulPartitionedCall/model_1/face_rpn_landmark_pred_stride16/BiasAdd/Add",
                      "StatefulPartitionedCall/model_1/tf.compat.v1.transpose_11/transpose",
                      "StatefulPartitionedCall/model_1/face_rpn_bbox_pred_stride8/BiasAdd/Add",
                      "StatefulPartitionedCall/model_1/face_rpn_landmark_pred_stride8/BiasAdd/Add"]

            output = q_det.get()
            dims = output.getAllLayers()
            dimensions = []
            nn_output = []
            for l in dims:
                dimensions.append(l.dims)
            # Reorder dimensions to match layers
            idx = [7, 1, 4, 8, 0, 3, 6, 2, 5]
            dimensions = [dimensions[i] for i in idx]
            i = 0
            for line in layers:
                data = (output.getLayerFp16(line))
                reshaped = np.array(data).reshape(dimensions[i][0], dimensions[i][1], dimensions[i][2],
                                                  dimensions[i][3])
                nn_output.append(reshaped)
                i += 1

            check = get_data(nn_output, image, im_info, im_scale, align=True)

            if check is None:
                # print("none")
                temp = getTime(time.time() - started)
                print("Total Run Time: {}".format(temp))
                # if not que.empty():
                #     que.get()
                # que[0] = temp
                pass
            else:
                plt.imsave(path + "/images/{}".format(timestr + ".png"), img_rgb)
                target_img, age_gender = check
                # toc = time.time()
                # print(toc - tic, "seconds - multiface")
                nn_data = dai.NNData()
                nn_data_age = dai.NNData()
                facial_data = []
                raw_stats = []
                for i in range(len(target_img)):
                    inputs = np.moveaxis(target_img[i], -1, 1)
                    nn_data.setLayer("0", inputs * 255)
                    landIn.send(nn_data)
                    land = landOut.get()
                    land_data = land.getFirstLayerFp16()
                    reshape = np.array(land_data).reshape(128)
                    datatype = np.array(reshape, np.float32)
                    facial_data.append(datatype)

                    # age = np.moveaxis(age_gender[i], -1, 1)
                    # nn_data_age.setLayer("0", age * 255)
                    # ageIn.send(nn_data_age)
                    # age = ageOut.get()
                    # age_data = age.getFirstLayerFp16()
                    # raw_stats.append(age_data)
                # stats = statistics(raw_stats)
                stats = ["empty", "empty", "empty"]
                sql(facial_data, target_img_path, database, time_stamp, stats, target_img, timestr, path)
                toc = time.time()
                print("Time to process frame: {}".format(getTime(toc - tic)))
                temp = getTime(time.time() - started)
                print("Total Run Time: {}".format(temp))
                # if not que.empty():
                #     que.get()
                # que.put(temp)
    # cv2.destroyAllWindows()

def starting(que, path, flip):
    pool = Process(target=cv_main, args=(que, path, flip,))
    pool.start()
    return pool


que = Queue(maxsize=1)
if __name__ == '__main__':
    finished = False
    flip = False
    path = getPath(__file__)
    runtime = ""
    while finished is False:
        time.sleep(1)
        pool = starting(que, path, flip)
        try:
            pool.join(timeout=0)
            while True:
                if pool.is_alive():
                    pass
                # elif not que.empty():
                #     tock = que.get()
                #     print("Finished in: {} seconds".format(tock-tick))
                #     finished = True
                #     break
                else:
                    if not que.empty():
                        runtime = que.get()
                        print("Ran for: {} before dying".format(runtime))
                    raise Exception("Thread died")
        except Exception:
            time.sleep(1)
            try:
                timestr = datetime.utcfromtimestamp((NTPClient().request(host='time.kku.ac.th', version=3, port='ntp')).tx_time)
            except:
                timestr = datetime.now()
            with open(path + "/log.txt", "a") as f:
                f.write("I crashed at {} and ran for {}\n".format(timestr, runtime))
            print("crashed at {}".format(timestr))
