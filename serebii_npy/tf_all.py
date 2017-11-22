import cv2
import numpy as np
import pylab
import tf_affine, tf_color, tf_rect
from tqdm import trange, tqdm
import sys


def tf(img):
    tfimgs = []
    # yellow rect : 1 * 2 * 2
    for size in trange(5, 6, desc="rect s"): #size
        for y in trange(16, 20, 3, desc="rect y"): #y
            for x in trange(16, 20, 3, desc="rect x"): #x
                rect_img = tf_rect.rect(img, x, y, size, size, (255, 255, 0))
                mask_img = (np.sum(rect_img, axis=2) != 0) * 255
                mask_img = cv2.merge((mask_img, mask_img, mask_img))
                mask_img = mask_img.astype(np.uint8)

                # rgb and hls : 3 * 4 * 3
                for rgb in range(3): #rgb
                    for i in range(20, 81, 20): #color
                        cimg = tf_color.color(rgb, rect_img, i, mask_img)
                        for j in range(-60, 1, 30): #hls
                            himg = tf_color.hls(cimg, j, mask_img)

                            # to 100x100
                            himg = cv2.resize(himg, (227, 227))

                            # affine : 3 * 3 * 3
                            for vr in range(10, 31, 10): #right
                                for vl in range(-10, 11, 10): #left
                                    for top in range(-10, 11, 10): #top
                                        p2 = np.float32([
                                            [50, i],
                                            [80 + vr, 80],
                                            [20 + vl, 80]
                                            ])
                                        amig = tf_affine.affine(himg, p2)
                                        tfimgs.append(amig)

    result = np.array(tfimgs)
    return result


for i, n in enumerate([233, 445, 797, 785, 130, 681]):
    print("index: {}, n: {}".format(i, n))
    img = tf_rect.readimg(n)
    result = tf(img)
    np.save("./tf/tf_227_{}_{}.npy".format(i, n), result)
