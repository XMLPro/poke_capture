import cv2
import numpy as np
import pylab
import tf_affine, tf_color, tf_rect
from tqdm import trange, tqdm
import sys


img = tf_rect.readimg(233)
cnt = 0
tfimgs = []

# yellow rect : 3 * 2 * 2
for size in trange(3, 6, desc="rect s"):
    for y in trange(16, 20, 3, desc="rect y"):
        for x in trange(16, 20, 3, desc="rect x"):
            rect_img = tf_rect.rect(img, x, y, size, size, (255, 255, 0))
            mask_img = (np.sum(rect_img, axis=2) != 0) * 255
            mask_img = cv2.merge((mask_img, mask_img, mask_img))
            mask_img = mask_img.astype(np.uint8)

            # rgb and hls : 3 * 5 * 3
            for rgb in trange(3, desc="rgb"):
                for i in trange(20, 81, 20, desc="col"):
                    cimg = tf_color.color(rgb, rect_img, i, mask_img)
                    for j in range(-60, 1, 30):
                        himg = tf_color.hls(cimg, j, mask_img)

                        # to 100x100
                        himg = cv2.resize(himg, (100, 100))

                        # affine : 5 * 5 * 5
                        for vr in range(0, 41, 10):
                            for vl in range(0, 41, 10):
                                for top in range(0, 41, 10):
                                    p2 = np.float32([
                                        [50, i],
                                        [80 - 20 + vr, 80],
                                        [20 + 20 - vl, 80]
                                        ])
                                    amig = tf_affine.affine(himg, p2)
                                    tfimgs.append(amig)

result = np.array(tfimgs)
np.save("./politf.npy", result)

            # cnt += 1
            # pylab.subplot(5 * 3, 5, cnt)
            # pylab.axis("off")
            # pylab.imshow(mask)
# pylab.show()
