import numpy as np
import chainer
import glob
from tqdm import tqdm
label = []
data = []

for path in glob.glob("tf/tf_227_*"):
    print(path)
    d = np.load(path)
    print(d.shape)
    length = d.shape[0]

    data.extend(d)
    label.extend([int(path[10])] * length)


data = np.array(data)
label = np.array(label, dtype=int)

data_n = data.shape[0]
np.save("./x_tfsix_227_{}.npy".format(data_n), data)
np.save("./t_tfsix_227_{}.npy".format(data_n), label)
