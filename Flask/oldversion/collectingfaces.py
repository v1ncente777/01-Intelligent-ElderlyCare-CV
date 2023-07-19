import cv2
import threading

from oldcare.facial import FaceUtil
from oldcare.audio import audioplayer
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import shutil
import time
from util import oss_upload


class CollectingFacesThread(threading.Thread):
    def __init__(self, absolute_path, uid):
        threading.Thread.__init__(self)
        self.id = uid
        self.absolute_path = absolute_path
        self.image_dir = absolute_path + 'images'
        self.audio_dir = absolute_path + 'audios'
        self.error = 0
        self.start_time = None
        self.limit_time = 2
        self.action_list = ['blink', 'open_mouth', 'smile', 'rise_head', 'bow_head',
                            'look_left', 'look_right']
        self.action_map = {'blink': '请眨眼', 'open_mouth': '请张嘴',
                           'smile': '请笑一笑', 'rise_head': '请抬头',
                           'bow_head': '请低头', 'look_left': '请看左边',
                           'look_right': '请看右边'}
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)  # set video width
        self.cam.set(4, 480)  # set video height
        self.face_util = FaceUtil()

    def run(self):
        counter = 0
        finish = True
        while True:
            counter += 1
            _, image = self.cam.read()
            if counter <= 10:  # 放弃前10帧
                continue
            image = cv2.flip(image, 1)

            if self.error == 1:
                end_time = time.time()
                difference = end_time - self.start_time
                print(difference)
                if difference >= self.limit_time:
                    self.error = 0

            face_location_list = self.face_util.get_face_location(image)
            for (left, top, right, bottom) in face_location_list:
                cv2.rectangle(image, (left, top), (right, bottom),
                              (0, 0, 255), 2)

            cv2.imshow('Collecting Faces', image)  # show the image
            # Press 'ESC' for exiting video
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                finish = False
                break

            face_count = len(face_location_list)
            if self.error == 0 and face_count == 0:  # 没有检测到人脸
                print('[WARNING] 没有检测到人脸')
                audioplayer.play_audio(os.path.join(self.audio_dir,
                                                    'no_face_detected.mp3'))
                self.error = 1
                self.start_time = time.time()
            elif self.error == 0 and face_count == 1:  # 可以开始采集图像了
                print('[INFO] 可以开始采集图像了')
                audioplayer.play_audio(os.path.join(self.audio_dir,
                                                    'start_image_capturing.mp3'))
                temp = self.collect()
                if temp == 0:
                    break
                else:
                    self.createdir()
                    self.error = 1
                    self.start_time = time.time()
            elif self.error == 0 and face_count > 1:  # 检测到多张人脸
                print('[WARNING] 检测到多张人脸')
                audioplayer.play_audio(os.path.join(self.audio_dir, 'multi_faces_detected.mp3'))
                self.error = 1
                self.start_time = time.time()
            else:
                pass

        # 结束
        if finish:
            print('[INFO] 采集完毕')
            audioplayer.play_audio(os.path.join(self.audio_dir, 'end_capturing.mp3'))
            oss_upload(self.absolute_path + 'images/' + self.id, self.id)

        # 释放全部资源
        self.cam.release()
        cv2.destroyAllWindows()

    def collect(self):

        for action in self.action_list:
            audioplayer.play_audio(os.path.join(self.audio_dir, action + '.mp3'))
            action_name = self.action_map[action]

            counter = 1
            for i in range(15):
                print('%s-%d' % (action_name, i))
                _, img_opencv = self.cam.read()
                img_opencv = cv2.flip(img_opencv, 1)
                origin_img = img_opencv.copy()  # 保存时使用

                face_location_list = self.face_util.get_face_location(img_opencv)
                face_count = len(face_location_list)
                for (left, top, right, bottom) in face_location_list:
                    cv2.rectangle(img_opencv, (left, top),
                                  (right, bottom), (0, 0, 255), 2)

                img_pil = Image.fromarray(cv2.cvtColor(img_opencv,
                                                       cv2.COLOR_BGR2RGB))

                draw = ImageDraw.Draw(img_pil)
                draw.text((int(img_opencv.shape[1] / 2), 30), action_name,
                          font=ImageFont.truetype('C:\\Windows\\Fonts\\STFANGSO.TTF', 40),
                          fill=(255, 0, 0))  # linux

                # 转换回OpenCV格式
                img_opencv = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

                cv2.imshow('Collecting Faces', img_opencv)  # show the image

                image_name = os.path.join(self.image_dir, self.id,
                                          action + '_' + str(counter) + '.jpg')
                cv2.imwrite(image_name, origin_img)

                if face_count == 0:  # 没有检测到人脸
                    print('[WARNING] 没有检测到人脸')
                    audioplayer.play_audio(os.path.join(self.audio_dir,
                                                        'no_face_detected.mp3'))
                    return 1
                elif face_count > 1:  # 检测到多张人脸
                    print('[WARNING] 检测到多张人脸')
                    audioplayer.play_audio(os.path.join(self.audio_dir,
                                                        'multi_faces_detected.mp3'))
                    return 1
                else:
                    pass

                # Press 'ESC' for exiting video
                k = cv2.waitKey(100) & 0xff
                if k == 27:
                    break
                counter += 1
        return 0

    def createdir(self):
        if os.path.exists(os.path.join(self.image_dir, self.id)):
            shutil.rmtree(os.path.join(self.image_dir, self.id), True)
        os.mkdir(os.path.join(self.image_dir, self.id))
