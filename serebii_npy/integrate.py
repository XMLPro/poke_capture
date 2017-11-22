import numpy as np
import chainer
import glob
label = []
data = []

data_n = 0
for path in glob.glob("tf/tf_227_*"):
    d = np.load(path)
    length = d.shape[0]
    data_n += length

    data.extend(d)
    label.extend([int(path[10])] * length)


data = np.array(data, dtype=np.float32)
label = np.array(label, dtype=int)
np.save("./x_tfsix_227_{}.npy".format(data_n), data)
np.save("./t_tfsix_227_{}.npy".format(data_n), label)
