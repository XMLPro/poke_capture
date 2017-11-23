import cv2
import numpy as np
import pylab


def to_black(img):
    return img[:, :, :3] * (img[:, :, 3:] / 255)
