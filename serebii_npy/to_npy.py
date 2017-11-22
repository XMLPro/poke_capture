from PIL import Image
import numpy as np
import cv2

filename = "../serebii/{}.png".format
targets = [233, 445, 797, 785, 130, 681]

data = []
for target in targets:
    img = Image.open(filename(target)).convert("RGB")
    img = np.asarray(img)
    img = cv2.resize(img, (100, 100))
    data.append(img.astype(np.float32) / 255.0)

result = np.array(data, dtype=np.float32).transpose((0, 3, 1, 2))
np.save("./six_serebii_100.npy", result)

