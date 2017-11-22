import cv2
import numpy as np
import pylab
import tf_affine, tf_color, tf_rect
from tqdm import trange, tqdm
import sys


img = tf_rect.readimg(233)
cnt = 0
tfimgs = []

# yellow rect : 3 * 5 * 5
for size in trange(3, 6, desc="rect s"):
    for y in trange(16, 26, 2, desc="rect y"):
        for x in trange(16, 26, 2, desc="rect x"):
            rect_img = tf_rect.rect(img, x, y, size, size, (255, 255, 0))
            mask_img = (np.sum(rect_img, axis=2) != 0) * 255
            mask_img = cv2.merge((mask_img, mask_img, mask_img))
            mask_img = mask_img.astype(np.uint8)

            # rgb and hls : 3 * 5 * 5
            for rgb in trange(3, desc="rgb"):
                for i in trange(20, 100, 20, desc="col"):
                    for j in range(-80, 20, 20):
                        cimg = tf_color.color(rgb, rect_img, i, mask_img)
                        cimg = tf_color.hls(cimg, j, mask_img)

                        # to 100x100
                        cimg = cv2.resize(cimg, (100, 100))

                        # affine : 8 * 8 * 10
                        for vr in range(0, 40, 5):
                            for vl in range(0, 40, 5):
                                for top in range(0, 50, 5):
                                    p2 = np.float32([
                                        [50, i],
                                        [80 - 20 + vr, 80],
                                        [20 + 20 - vl, 80]
                                        ])
                                    amig = tf_affine.affine(cimg, p2)
                                    tfimgs.append(amig)

result = np.array(tfimgs)
np.save("./politf.npy", result)

            # cnt += 1
            # pylab.subplot(5 * 3, 5, cnt)
            # pylab.axis("off")
            # pylab.imshow(mask)
# pylab.show()
