import cv2
import numpy as np
import pylab

p1 = np.float32([
    [50, 20],
    [80, 80],
    [20, 80]
    ])

imgs = np.load("six_serebii_100.npy")

img = imgs[0]
img = img.transpose((1, 2, 0))

images = []

cnt = 0
for vr in range(0, 40, 5):
    for vl in range(0, 40, 5):
        for i in range(0, 50, 5):
            cnt += 1
            p2 = np.float32([
                [50, i],
                [80 - 20 + vr, 80],
                [20 + 20 - vl, 80]
                ])

            w, h, _ = img.shape
            M = cv2.getAffineTransform(p1, p2)
            tfimg = cv2.warpAffine(img, M, (w, h))
            images.append(tfimg)

            # if cnt % 2:
            #     pylab.axis("off")
            #     pylab.subplot(32, 10, cnt // 2 + 1)
            #
            # for p in p1:
            #     pylab.plot(p[0], p[1], 'ro')
            #
            # for p in p2:
            #     pylab.plot(p[0], p[1], 'bo')
#             pylab.imshow(tfimg)
# pylab.show()
np.save("./trans_640.npy", np.array(images, dtype=np.float32))
