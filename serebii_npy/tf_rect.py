import cv2
import numpy as np
import pylab
from PIL import Image


def readimg(number):
    img = Image.open("../serebii/{}.png".format(number))
    img = img.convert("RGB")
    return np.asarray(img)


def rect(img, x, y, w, h, color):
    img = np.array(img)
    img[y:y + h, x:x + w, :] = color
    return img


if __name__ == '__main__':
    cnt = 0
    for size in range(3, 6):
        for y in range(16, 26, 2):
            for x in range(16, 26, 2):
                cnt += 1
                img = readimg(233)
                img = rect(img, x, y, size, size, (255, 255, 0))
                pylab.subplot(5 * 3, 5, cnt)
                pylab.axis("off")
                pylab.imshow(img)
    pylab.show()
