# -*- coding: utf-8 -*-
'''
摔倒检测模型主程序

用法：
python testingfalldetection.py
python testingfalldetection.py --filename tests/corridor_01.avi
'''

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import time
import argparse

# 传入参数
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=False, default = '',
	help="")
args = vars(ap.parse_args())
input_video = args['filename']

# 控制陌生人检测
fall_timing = 0 # 计时开始
fall_start_time = 0 # 开始时间
fall_limit_time = 1 # if >= 1 seconds, then he/she falls.

# 全局变量
model_path = 'models/fall_detection.hdf5'

# 全局常量
TARGET_WIDTH = 64
TARGET_HEIGHT = 64

# 初始化摄像头
if not input_video:
	vs = cv2.VideoCapture(0)
	time.sleep(2)
else:
	vs = cv2.VideoCapture(input_video)

# 加载模型
model = load_model(model_path)
    
print('[INFO] 开始检测是否有人摔倒...')
# 不断循环
counter = 0
while True:
    counter += 1
    # grab the current frame
    (grabbed, image) = vs.read()

	# if we are viewing a video and we did not grab a frame, then we
	# have reached the end of the video
    if input_video and not grabbed:
        break
    
    if not input_video:
        image = cv2.flip(image, 1)
        
    roi = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
        
    # determine facial expression
    (fall, normal) = model.predict(roi)[0]
    label = "Fall (%.2f)" %(fall) if fall > normal else "Normal (%.2f)" %(normal)
    
    # display the label and bounding box rectangle on the output frame
    cv2.putText(image, label, (image.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    cv2.imshow('Fall detection', image)
    
    # Press 'ESC' for exiting video
    k = cv2.waitKey(1) & 0xff 
    if k == 27:
        break


vs.release()
cv2.destroyAllWindows()
