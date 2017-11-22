import cv2
import numpy as np
import pylab
from PIL import Image


def readimg(number):
    img = Image.open("../serebii/{}.png".format(number))
    gray_img = img.convert("L")
    img = img.convert("RGB")
    img = np.array(img)
    return img, gray_img


def mask(gray_img):
    _, mask = cv2.threshold(np.asarray(gray_img), 0, 255, cv2.THRESH_BINARY)
    return cv2.merge((mask, mask, mask))


def color(target_color, img, value, mask):
    img = np.array(img)
    reshaped = img.reshape((32 * 32, 3))
    for target in reshaped:
        target[target_color] = max(0, min(255.0, target[target_color] + value))
    return cv2.bitwise_and(img, mask)
color.r = 0
color.g = 1
color.b = 2


def hsv(img, value, mask):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for target in hsv_img.reshape((32 * 32, 3)):
        target[2] = max(0, min(255.0, target[2] + value))
    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return cv2.bitwise_and(hsv_img, mask)


def hls(img, value, mask):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    for target in img.reshape((32 * 32, 3)):
        target[1] = max(0, min(255.0, target[1] + value))
    img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    return cv2.bitwise_and(img, mask)


if __name__ == '__main__':
    # imgs = np.load("six_serebii_100.npy")
    # img = imgs[0]
    # img = img.transpose((1, 2, 0))

    maskimg = mask(gray_img)
    cnt = 0
    for rgb in range(3):
        for i in range(20, 100, 20):
            for j in range(-80, 20, 20):
                cnt += 1
                bimg = color(rgb, img, i, maskimg)
                bimg = hls(bimg, j, maskimg)
                pylab.subplot(12, 5, cnt)
                pylab.axis("off")
                pylab.imshow(bimg)
    pylab.show()
