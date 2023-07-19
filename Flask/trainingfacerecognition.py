# -*- coding: utf-8 -*-

'''
训练人脸识别模型
'''

# import the necessary packages
from imutils import paths
from oldcare.facial import FaceUtil
from util import oss_upload_model, oss_download_image

# global variable
dataset_path = 'images'
output_encoding_file_path = 'models/face_recognition_hog.pickle'
output_version_path = 'models/model_version.txt'


def train_face():
    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    image_paths = list(paths.list_images(dataset_path))

    if len(image_paths) == 0:
        print('[ERROR] no images to train.')
        return 'error'
    else:
        #oss_download_image()
        faceutil = FaceUtil()
        print("[INFO] training face embeddings...")
        faceutil.save_embeddings(image_paths, output_encoding_file_path)
        #oss_upload_model(output_encoding_file_path, output_version_path)


#train_face()