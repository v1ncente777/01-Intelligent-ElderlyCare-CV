import shutil

import cv2
from threading import Thread
from oldcare.facial import FaceUtil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import time
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from oldcare.utils import fileassistant
import imutils
from util import http_post, oss_upload

Emotion_List = ['Angry', 'Digust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


class StrangersExpression:
    def __init__(self, url, input_video=False):
        self.url = url
        # 得到当前时间
        self.vs = None
        self.input_video = input_video
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print('[INFO] %s 陌生人检测程序和表情检测程序启动了.' % current_time)
        print('[INFO] 开始检测陌生人和表情...')

        # 全局变量
        self.facial_recognition_model_path = 'models/face_recognition_hog.pickle'
        self.emotion_model_path = 'models/fer2013_mini_XCEPTION.33-0.65.hdf5'

        self.output_stranger_path = 'supervision/strangers'
        self.output_smile_path = 'supervision/smile'

        if os.path.exists(self.output_stranger_path):
            shutil.rmtree(self.output_stranger_path, True)
        os.mkdir(self.output_stranger_path)
        if os.path.exists(self.output_smile_path):
            shutil.rmtree(self.output_smile_path, True)
        os.mkdir(self.output_smile_path)

        self.people_info_path = 'info/people_info.csv'
        self.facial_expression_info_path = 'info/facial_expression_info.csv'

        # 全局常量
        self.FACIAL_EXPRESSION_TARGET_WIDTH = 28
        self.FACIAL_EXPRESSION_TARGET_HEIGHT = 28

        self.VIDEO_WIDTH = 640
        self.VIDEO_HEIGHT = 480

        self.ANGLE = 20

        self.strangers_timing = 0  # 计时开始
        self.strangers_start_time = 0  # 开始时间
        self.strangers_limit_time = 2  # if >= 2 seconds, then he/she is a stranger.
        self.strangers_resend_time = 60  # if >= 60 seconds, then resend message.
        self.stranger_exist = False

        self.facial_expression_timing = 0  # 计时开始
        self.facial_expression_start_time = 0  # 开始时间
        self.facial_expression_limit_time = 2  # if >= 2 seconds, he/she is smiling

        self.id_card_to_name, self.id_card_to_type = fileassistant.get_people_info(self.people_info_path)
        self.facial_expression_id_to_name = fileassistant.get_facial_expression_info(self.facial_expression_info_path)

        if not input_video:
            self.vs = cv2.VideoCapture(0)
            time.sleep(2)
        else:
            self.vs = cv2.VideoCapture(input_video)

        # 初始化人脸识别模型
        self.face_util = FaceUtil(self.facial_recognition_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.emotion_window = []

    def get_frame(self):
        # grab the current frame
        (grabbed, frame) = self.vs.read()

        if grabbed:
            if not self.input_video:
                frame = cv2.flip(frame, 1)

            frame = imutils.resize(frame, width=self.VIDEO_WIDTH, height=self.VIDEO_HEIGHT)  # 压缩，加快识别速度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale，表情识别

            face_location_list, names = self.face_util.get_face_location_and_name(frame)

            # 得到画面的四分之一位置和四分之三位置，并垂直划线
            one_fourth_image_center = (int(self.VIDEO_WIDTH / 4), int(self.VIDEO_HEIGHT / 4))
            three_fourth_image_center = (int(self.VIDEO_WIDTH / 4 * 3), int(self.VIDEO_HEIGHT / 4 * 3))

            cv2.line(frame, (one_fourth_image_center[0], 0),
                     (one_fourth_image_center[0], self.VIDEO_HEIGHT),
                     (0, 255, 255), 1)
            cv2.line(frame, (three_fourth_image_center[0], 0),
                     (three_fourth_image_center[0], self.VIDEO_HEIGHT),
                     (0, 255, 255), 1)

            # 处理每一张识别到的人脸
            for ((left, top, right, bottom), name) in zip(face_location_list, names):
                # 将人脸框出来
                rectangle_color = (0, 0, 255)
                if self.id_card_to_type[name] == 'old_people':
                    rectangle_color = (0, 0, 128)
                elif self.id_card_to_type[name] == 'employee':
                    rectangle_color = (255, 0, 0)
                elif self.id_card_to_type[name] == 'volunteer':
                    rectangle_color = (0, 255, 0)
                else:
                    pass
                cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, 2)

                # 陌生人检测逻辑
                if 'Unknown' in names:  # alert
                    if self.strangers_timing == 0:  # just start timing
                        self.strangers_timing = 1
                        self.strangers_start_time = time.time()
                    else:  # already started timing
                        strangers_end_time = time.time()
                        difference = strangers_end_time - self.strangers_start_time

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))

                        if difference < self.strangers_limit_time:
                            print('[INFO] %s, 房间, 陌生人仅出现 %.1f 秒. 忽略.' % (current_time, difference))
                        else:  # strangers appear
                            event_desc = '陌生人出现!!!'
                            event_location = '房间'
                            print('[EVENT] %s, 房间, 陌生人出现!!!' % current_time)

                            # insert into database
                            # command = '%s inserting.py --event_desc %s --event_type 2 --event_location %s' % (
                            # python_path, event_desc, event_location)
                            # p = subprocess.Popen(command, shell=True)
                            # def http_post(url, event_type, event_location, event_desc, old_people_id=None):
                            if not self.stranger_exist:
                                thread = Thread(target=http_post, args=(self.url, 2, event_location, event_desc))
                                thread.start()
                                cv2.imwrite(os.path.join(self.output_stranger_path,
                                                         'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                            frame)  # snapshot
                                self.stranger_exist = True
                            if self.stranger_exist and difference > self.strangers_resend_time:
                                thread = Thread(target=http_post, args=(self.url, 2, event_location, event_desc))
                                thread.start()
                                cv2.imwrite(os.path.join(self.output_stranger_path,
                                                         'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                            frame)  # snapshot
                                self.strangers_start_time = time.time()

                            # 开始陌生人追踪
                            unknown_face_center = (int((right + left) / 2),
                                                   int((top + bottom) / 2))

                            cv2.circle(frame, (unknown_face_center[0],
                                               unknown_face_center[1]), 4, (0, 255, 0), -1)

                            direction = ''
                            # face locates too left, servo need to turn right,
                            # so that face turn right as well
                            if unknown_face_center[0] < one_fourth_image_center[0]:
                                direction = 'right'
                            elif unknown_face_center[0] > three_fourth_image_center[0]:
                                direction = 'left'

                            # adjust to servo
                            if direction:
                                print('摄像头需要 turn %s %d 度' % (direction, self.ANGLE))

                else:  # everything is ok
                    self.strangers_timing = 0
                    self.stranger_exist = False

                # 表情检测逻辑
                # 如果不是陌生人，且对象是老人
                if name != 'Unknown' and self.id_card_to_type[name] == 'old_people':
                #if name == 'Unknown':
                    # 表情检测逻辑
                    roi = gray[top:bottom, left:right]
                    # roi = cv2.resize(roi, (self.FACIAL_EXPRESSION_TARGET_WIDTH,
                    #                        self.FACIAL_EXPRESSION_TARGET_HEIGHT))
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # determine facial expression
                    Emotion_id_list = self.emotion_classifier.predict(roi)[0]
                    Emotion_id_list = Emotion_id_list.tolist()
                    facial_expression_label = Emotion_List[Emotion_id_list.index(max(Emotion_id_list))]

                    if facial_expression_label == 'Happy':  # alert
                        if self.facial_expression_timing == 0:  # just start timing
                            self.facial_expression_timing = 1
                            self.facial_expression_start_time = time.time()
                        else:  # already started timing
                            facial_expression_end_time = time.time()
                            difference = facial_expression_end_time - self.facial_expression_start_time

                            current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                         time.localtime(time.time()))
                            if difference < self.facial_expression_limit_time:
                                print('[INFO] %s, 房间, %s仅笑了 %.1f 秒. 忽略.' % (
                                    current_time, self.id_card_to_name[name], difference))
                            else:  # he/she is really smiling
                                event_desc = '%s正在笑' % (self.id_card_to_name[name])
                                event_location = '房间'
                                print('[EVENT] %s, 房间, %s正在笑.' % (current_time, self.id_card_to_name[name]))
                                cv2.imwrite(os.path.join(self.output_smile_path,
                                                         'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                            frame)  # snapshot

                                # insert into database
                                # command = '%s inserting.py --event_desc %s --event_type 0 --event_location %s --old_people_id %d' % (
                                # python_path, event_desc, event_location, int(name))
                                # p = subprocess.Popen(command, shell=True)
                                thread = Thread(target=http_post, args=(self.url, 0, event_location, event_desc, name))
                                thread.start()

                    else:  # everything is ok
                        self.facial_expression_timing = 0

                else:  # 如果是陌生人，则不检测表情
                    facial_expression_label = ''

                # 人脸识别和表情识别都结束后，把表情和人名写上
                # (同时处理中文显示问题)
                img_PIL = Image.fromarray(cv2.cvtColor(frame,
                                                       cv2.COLOR_BGR2RGB))

                draw = ImageDraw.Draw(img_PIL)
                final_label = self.id_card_to_name[name] + ': ' + self.facial_expression_id_to_name[
                    facial_expression_label] if facial_expression_label else self.id_card_to_name[name]
                draw.text((left, top - 30), final_label,
                          font=ImageFont.truetype('C:\\Windows\\Fonts\\STFANGSO.TTF', 40),
                          fill=(255, 0, 0))  # linux

                # 转换回OpenCV格式
                frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

            # show our detected faces along with smiling/not smiling labels
            # cv2.imshow("Checking Strangers and Ole People's Face Expression",
            #            frame)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            return None
        # cleanup the camera and close any open windows

    def __del__(self):
        self.vs.release()
        oss_upload(self.output_stranger_path)
        oss_upload(self.output_smile_path)
        cv2.destroyAllWindows()
