import cv2
import numpy as np
import pylab
from PIL import Image

import tf_rect
import tf_color


def background(i, r):
    img = Image.open("../backs/back{}.jpeg".format(i))
    img = img.convert("RGB")
    img = np.asarray(img)
    img = cv2.resize(img, (50, 50))

    center = (img.shape[0] // 2, img.shape[1] // 2)
    rotMat = cv2.getRotationMatrix2D(center, r, 1.0)    
    rotated = cv2.warpAffine(img, rotMat, img.shape[0:2], flags=cv2.INTER_LINEAR)
    return rotated[25 - 32//2 : 25 + 32//2, 25 - 32//2 : 25 + 32//2]


def merge(back, img, mask):
    im0 = cv2.bitwise_and(img, mask)
    im1 = cv2.bitwise_and(back, cv2.bitwise_not(mask))
    return cv2.bitwise_or(im0, im1)



if __name__ == "__main__":
    img, gray = tf_color.readimg(233)
    mask = tf_color.mask(gray)
    back = background(5, 19)
    result = merge(back, img, mask)
