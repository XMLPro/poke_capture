import cv2
import numpy as np
import pylab
import tf_affine, tf_color, tf_rect, tf_back
from tqdm import trange, tqdm
import sys
import random


def tf(n):
    tfimgs = []
    # yellow rect : 1 * 2 * 2
    # for size in trange(5, 6, desc="rect s"): #size
    #     for y in trange(16, 20, 3, desc="rect y"): #y
    #         for x in trange(16, 20, 3, desc="rect x"): #x
    #             rect_img = tf_rect.rect(img, x, y, size, size, (255, 255, 0))
    #             mask_img = (np.sum(rect_img, axis=2) != 0) * 255
    #             mask_img = cv2.merge((mask_img, mask_img, mask_img))
    #             mask_img = mask_img.astype(np.uint8)
    #
    #             # rgb and hls : 3 * 4 * 3
    #             for rgb in range(3): #rgb
    #                 for i in range(20, 81, 20): #color
    #                     cimg = tf_color.color(rgb, rect_img, i, mask_img)
    #                     for j in range(-60, 1, 30): #hls
    #                         himg = tf_color.hls(cimg, j, mask_img)
    #
    #                         # to 100x100
    #                         himg = cv2.resize(himg, (227, 227))

    img, gray = tf_color.readimg(n)
    # mask = tf_color.mask(gray)
    # img = tf_back.merge(back, img, mask)

    # affine : 4 * 4 * 4
    for vr in trange(-4, 3, 2, desc="right"): #right
        for vl in range(-3, 4, 2): #left
            for top in range(-4, 6, 3): #top
                p2 = np.float32([
                    [16, 6 + top],
                    [26 + vr, 26],
                    [6 + vl, 26]
                    ])
                amig = tf_affine.affine(img, p2)
                agray = tf_affine.affine(gray, p2)
                mask = tf_color.mask(agray)
                back = tf_back.background(random.randint(0, 5), random.randint(0, 360))
                amig = tf_back.merge(back, amig, mask)

                if random.random() < 0.2:
                    amig = tf_rect.rect(amig, 20 + random.randint(0, 4), 20 + random.randint(0, 4), 6, 6, (255, 255, 0))
                for i in trange(5, desc="lg"):
                    lgimg = lgpy.blur(amig, i, 32).astype(np.uint8)
                    tfimgs.append(lgimg.transpose(2, 0, 1))

    result = np.array(tfimgs)
    return result


from lgfilter import lgpy
x_data = []
t_data = []
for i, n in enumerate([233, 445, 797, 785, 130, 681]):
    print("index: {}, n: {}".format(i, n))
    result = tf(n)
    x_data.extend(result)
    t_data.extend([i] * len(result))
    # for index, x in enumerate(result[::4], 1):
    #     pylab.subplot(10, 8, index)
    #     pylab.axis("off")
    #     pylab.imshow(x)
    # pylab.show()
np.save("./data/x_tflg_{}.npy".format(len(t_data)), np.array(x_data, dtype=np.float32) / 255.0)
np.save("./data/t_tflg_{}.npy".format(len(t_data)), np.array(t_data, dtype=np.uint8))
