# from django.db import models
#
# # Create your models here.
#
#
# class Pose(models.Model):
#     image = models.ImageField(upload_to='pose/', null=False)
from os.path import dirname, join

from django.db import models
from .utils import output_keypoints, output_keypoints_with_lines
from PIL import Image
import numpy as np
from io import BytesIO
from django.core.files.base import ContentFile

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


# Create your models here.
class Pose(models.Model):
    image = models.ImageField(upload_to='image/', blank=True, null=True)

    def save(self, *args, **kwargs):
        # 이미지 열기
        pil_img = Image.open(self.image)

        # np array 변환
        cv_img = np.array(pil_img)
        cv_img = rgba2rgb(cv_img)
        # print(cv_img.shape)

        # cv_img = cv_img.reshape(-1, -1, -1)
        # 모델 위치
        # protoFile_body_25 = os.path.join(BASE_DIR, "model/pose_deploy.prototxt")
        # weightsFile_body_25 = os.path.join(BASE_DIR, "model/pose_iter_584000.caffemodel")
        protoFile_body_25 = "C:/Users/ohjoonhoo/Downloads/turtleneck/model/pose_deploy.prototxt"
        weightsFile_body_25 = "C:/Users/ohjoonhoo/Downloads/turtleneck/model/pose_iter_584000.caffemodel"
        # protoFile_body_25 = join(dirname(__file__), "model/pose_deploy.prototxt")
        # weightsFile_body_25 = join(dirname(__file__), "model/pose_iter_584000.caffemodel")

        BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                              5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                              10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                              15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                              20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel",
                              25: "Background"}

        POSE_PAIRS_BODY_25 = [[5, 18], [2, 17]] # [0, 2], [0, 5],

        frame_man = output_keypoints(frame=cv_img, proto_file=protoFile_body_25, weights_file=weightsFile_body_25,
                                     threshold=0.1, model_name='', BODY_PARTS=BODY_PARTS_BODY_25)
        img = output_keypoints_with_lines(POSE_PAIRS=POSE_PAIRS_BODY_25, frame=frame_man)

        # convert back to pil image
        im_pil = Image.fromarray(img)

        # 저장
        buffer = BytesIO()
        im_pil.save(buffer, format='png')
        image_png = buffer.getvalue()

        self.image.save(str(self.image), ContentFile(image_png), save=False)

        super().save(*args, **kwargs)
