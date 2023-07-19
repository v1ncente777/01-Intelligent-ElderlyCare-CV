from threading import Thread
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import time
import subprocess
import argparse
from util import http_post


class CheckFall:
    def __init__(self, url, input_video=False):
        self.url = url

        # 得到当前时间
        self.vs = None
        self.input_video = input_video
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print('[INFO] %s 摔倒检测程序启动了.' % current_time)

        # 全局变量
        self.model_path = 'models/fall_detection.hdf5'
        self.output_fall_path = 'supervision/fall'

        self.TARGET_WIDTH = 64
        self.TARGET_HEIGHT = 64

        # 全局常量
        self.fall_timing = 0  # 计时开始
        self.fall_start_time = 0  # 开始时间
        self.fall_limit_time = 1  # if >= 1 seconds, then he/she falls.
        self.fall_resend_time = 60
        self.fall_exist = False

        if not input_video:
            self.vs = cv2.VideoCapture(0)
            time.sleep(2)
        else:
            self.vs = cv2.VideoCapture(input_video)

        # 加载模型
        self.model = load_model(self.model_path)

        print('[INFO] 开始检测是否有人摔倒...')

    def get_frame(self):

        # grab the current frame
        (grabbed, image) = self.vs.read()

        if grabbed:
            if not self.input_video:
                image = cv2.flip(image, 1)

            roi = cv2.resize(image, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # determine facial expression
            (fall, normal) = self.model.predict(roi)[0]
            label = "Fall (%.2f)" % fall if fall < normal else "Normal (%.2f)" % normal

            # display the label and bounding box rectangle on the output frame
            cv2.putText(image, label, (image.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            if fall > normal:
                if self.fall_timing == 0:  # just start timing
                    self.fall_timing = 1
                    self.fall_start_time = time.time()
                else:  # already started timing
                    fall_end_time = time.time()
                    difference = fall_end_time - self.fall_start_time

                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))

                    if difference < self.fall_limit_time:
                        print('[INFO] %s, 走廊, 摔倒仅出现 %.1f 秒. 忽略.' % (current_time, difference))
                    else:  # strangers appear
                        event_desc = '有人摔倒!!!'
                        event_location = '走廊'
                        print('[EVENT] %s, 走廊, 有人摔倒!!!' % current_time)
                        # insert into database
                        # command = '%s inserting.py --event_desc %s --event_type 3 --event_location %s' % (
                        # python_path, event_desc, event_location)
                        # p = subprocess.Popen(command, shell=True)
                        if not self.fall_exist:
                            thread = Thread(target=http_post, args=(self.url, 3, event_location, event_desc))
                            thread.start()
                            cv2.imwrite(os.path.join(self.output_fall_path,
                                                     'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                        image)  # snapshot
                            self.fall_exist = True
                        if self.fall_exist and difference > self.fall_resend_time:
                            thread = Thread(target=http_post, args=(self.url, 3, event_location, event_desc))
                            thread.start()
                            cv2.imwrite(os.path.join(self.output_fall_path,
                                                     'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                        image)  # snapshot
                            self.fall_start_time = time.time()

                        # show our detected faces along with smiling/not smiling labels
            # cv2.imshow("Checking Strangers and Ole People's Face Expression",
            #            frame)
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            return None
        # cleanup the camera and close any open windows

    def __del__(self):
        self.vs.release()
        cv2.destroyAllWindows()
