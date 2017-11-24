import numpy
import chainer
from chainer import links as L, functions as F
from chainer.training import extensions
import pylab
import cv2
import numpy as np

device = 1

labelname = [ 233, 445, 797, 785, 130, 681, ]

# x_data = np.load("lg/xtdata/x_pokedata1000.npy").astype(np.float32) / 255
# t_data = np.load("lg/xtdata/t_pokedata1000.npy").astype(np.int32)

print("load data...")
x_data = np.load("data/x_tflg_gray_3840.npy").astype(np.float32)
print("load label...")
t_data = np.load("data/t_tflg_gray_3840.npy").astype(np.int32)

# dec = 5
# x_data = x_data[::dec]
# t_data = t_data[::dec]
#
# x_data = x_data.transpose(0, 3, 1, 2)


class IMAGE(chainer.Chain):
    def __init__(self):
        super().__init__(
                # # (227 - 11) / 4 + 1 => 55
                # c1=L.Convolution2D(3, 32, 11, stride=4),
                # # pooling 4, stride=4 | (55 - 5) / 5 + 1 => 11
                # b1=L.BatchNormalization(32),
                # # 27 - 4 + 1 24
                # oc1=L.Convolution2D(96, 32, 4),
                # # 24 - 3 + 1 22
                # oc2=L.Convolution2D(32, 16, 3),
                # 7744
                # l1=L.Linear(3872, 1024),
                # l2=L.Linear(1024, 256),
                # l3=L.Linear(256, 6),

                # # (227 - 11) / 4 + 1 => 55
                # c1=L.Convolution2D(3, 96, 11, stride=4),
                # # pooling 3, stride=2 | 27
                # b1=L.BatchNormalization(96),
                # # # (27 + 2*2 - 5) / 1 => 27
                # c2=L.Convolution2D(96, 256, 5, pad=2),
                # # # pooling 3, stride=2 | 13
                # b2=L.BatchNormalization(256),
                # # # (13 + 1*2 - 3) / 1 => 13
                # c3=L.Convolution2D(256, 256, 3, pad=1),
                # # # (13 + 1*2 - 3) / 1 => 13
                # c4=L.Convolution2D(256, 256, 3, pad=1),
                # # # (13 + 1*2 - 3) / 1 => 13
                # # c5=L.Convolution2D(256, 256, 3, pad=1),
                # # # pooling 3, stide=2 | 6
                # l1=L.Linear(256 * 6 * 6, 4096),
                # l2=L.Linear(4096, 1024),
                # l3=L.Linear(1024, 6),
                l1=L.Linear(1024, 1024),
                l2=L.Linear(1024, 144),
                l3=L.Linear(144, 6),
                )

    def show(self, h, size, data=None):
        img = h.data[0]
        ch = len(img)
        for i, v in enumerate(img, 1):
            pylab.subplot(int(ch ** 0.5) + 1, int(ch ** 0.5) + 1, i)
            pylab.axis('off')
            pylab.imshow(v)
        # pylab.show()
        pylab.savefig("graph/conv_{}_{}.png".format(data[0], data[1]))

    def __call__(self, x, data=None):
        cnt = 0
        h = x
        # h = F.max_pooling_2d(F.relu(self.b1(self.c1(h))), 3, stride=2)
        # # self.show(h, h.shape[-1], data=[data, cnt]); cnt += 1
        # h = F.max_pooling_2d(F.relu(self.b2(self.c2(h))), 3, stride=2)
        # # self.show(h, h.shape[-1], data=[data, cnt]); cnt += 1
        # # h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.c1(h))), 3, stride=2)
        # # h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.c2(h))), 3, stride=2)
        # h = F.relu(self.c3(h))
        # # self.show(h, h.shape[-1], data=[data, cnt]); cnt += 1
        # h = F.relu(self.c4(h))
        # # self.show(h, h.shape[-1], data=[data, cnt]); cnt += 1
        # # h = F.relu(self.c5(h))
        # h = F.max_pooling_2d(h, 3, stride=2)
        # # self.show(h, h.shape[-1], data=[data, cnt]); cnt += 1
        # # h = F.relu(self.l1(h))
        # h = F.dropout(F.relu(self.l1(h)))
        # # pylab.subplot(2, 6, data * 2 + 1)
        # # pylab.axis("off")
        # # pylab.imshow(h.data.reshape((64, 64)).astype(np.uint8) * 255)
        # # pylab.savefig("graph/fc_{}_{}.png".format(data, 0))
        # # h = F.relu(self.l2(h))
        # h = F.dropout(F.relu(self.l2(h)))
        # # pylab.subplot(2, 6, data * 2 + 2)
        # # pylab.axis("off")
        # # pylab.imshow(h.data.reshape((32, 32)).astype(np.uint8) * 255)
        # # pylab.savefig("graph/fc_{}_{}.png".format(data, 1))
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        return self.l3(h)

# weight_ = IMAGE()
# weight = L.Classifier(weight_, lossfun=F.softmax_cross_entropy, accfun=F.accuracy)
# weight_path = ""
# chainer.serializers.load_npz(weight_path, weight)
# class IMAGE_TRANS(chainer.Chain):
#     def __init__(self):
#         super().__init__(
#                 # (227 - 11) / 4 + 1 => 55
#                 c1=L.Convolution2D(3, 96, 11, stride=4,
#                     initialW=weight.predictor.c1.W.data,
#                     initial_bias=weight.predictor.c1.b.data),
#                 # pooling 3, stride=2 | 27
#                 b1=L.BatchNormalization(96,
#                     initial_gamma=weight.predictor.b1.gamma.data,
#                     initial_beta=weight.predictor.b1.beta.data),
#                 # # (27 + 2*2 - 5) / 1 => 27
#                 c2=L.Convolution2D(96, 256, 5, pad=2,
#                     initialW=weight.predictor.c2.W.data,
#                     initial_bias=weight.predictor.c2.b.data),
#                 # # pooling 3, stride=2 | 13
#                 b2=L.BatchNormalization(256,
#                     initial_gamma=weight.predictor.b2.gamma.data,
#                     initial_beta=weight.predictor.b2.beta.data),
#                 # # (13 + 1*2 - 3) / 1 => 13
#                 c3=L.Convolution2D(256, 256, 3, pad=1,
#                     initialW=weight.predictor.c3.W.data,
#                     initial_bias=weight.predictor.c3.b.data),
#                 # # (13 + 1*2 - 3) / 1 => 13
#                 c4=L.Convolution2D(256, 256, 3, pad=1,
#                     initialW=weight.predictor.c4.W.data,
#                     initial_bias=weight.predictor.c4.b.data),
#                 # # pooling 3, stide=2 | 6
#                 l1=L.Linear(256 * 6 * 6, 4096),
#                 l2=L.Linear(4096, 1024),
#                 l3=L.Linear(1024, 6),
#                 )
#
#     def show(self, h, size, data=None):
#         img = h.data[0]
#         ch = len(img)
#         for i, v in enumerate(img, 1):
#             pylab.subplot(int(ch ** 0.5) + 1, int(ch ** 0.5) + 1, i)
#             pylab.axis('off')
#             pylab.imshow(v)
#         # pylab.show()
#         pylab.savefig("graph/conv_{}_{}.png".format(data[0], data[1]))
#
#     def __call__(self, x, data=None):
#         cnt = 0
#         with chainer.no_backprop_mode():
#             h = x
#             h = F.max_pooling_2d(F.relu(self.b1(self.c1(h))), 3, stride=2)
#             # self.show(h, h.shape[-1], data=[data, cnt]); cnt += 1
#             h = F.max_pooling_2d(F.relu(self.b2(self.c2(h))), 3, stride=2)
#             # self.show(h, h.shape[-1], data=[data, cnt]); cnt += 1
#             h = F.relu(self.c3(h))
#             # self.show(h, h.shape[-1], data=[data, cnt]); cnt += 1
#             h = F.relu(self.c4(h))
#             # self.show(h, h.shape[-1], data=[data, cnt]); cnt += 1
#             h = F.max_pooling_2d(h, 3, stride=2)
#             # self.show(h, h.shape[-1], data=[data, cnt]); cnt += 1
#         h = F.relu(self.l1(h))
#         # h = F.dropout(F.relu(self.l1(h)))
#         # pylab.subplot(2, 6, data * 2 + 1)
#         # pylab.axis("off")
#         # pylab.imshow(h.data.reshape((64, 64)).astype(np.uint8) * 255)
#         # pylab.savefig("graph/fc_{}_{}.png".format(data, 0))
#         h = F.relu(self.l2(h))
#         # h = F.dropout(F.relu(self.l2(h)))
#         # pylab.subplot(2, 6, data * 2 + 2)
#         # pylab.axis("off")
#         # pylab.imshow(h.data.reshape((32, 32)).astype(np.uint8) * 255)
#         # pylab.savefig("graph/fc_{}_{}.png".format(data, 1))
#         return self.l3(h)


optimizer = chainer.optimizers.Adam()
imodel = IMAGE()
model = L.Classifier(imodel, lossfun=F.softmax_cross_entropy, accfun=F.accuracy)
optimizer.setup(model)

import os
model_path = "./tf_poke_gray.npz"
cnt = 0
from PIL import Image
from lgfilter import lgpy
from tqdm import tqdm
if os.path.exists(model_path):
    with chainer.no_backprop_mode():
        chainer.serializers.load_npz(model_path, model)
        # target = model.predictor.c1.W.data
        # print(target.shape)
        # print(np.max(target - np.min(target)))
        # for index, c in enumerate(target, 1):
        #     c -= np.min(c)
        #     c *= 255
        #     c = c.astype(np.uint8)
        #     for cc in c:
        #         pylab.subplot(target.shape[0] ** 0.5 + 1, target.shape[0] ** 0.5 + 1, index)
        #         pylab.axis("off")
        #         pylab.imshow(cc)
        # pylab.show()
        # - - -
        x_test = np.load("./six_small_orange_70x70.npy")
        # for i, v in enumerate(tqdm(x_test)):
        #     img = cv2.resize(v, (227, 227), interpolation=cv2.INTER_CUBIC).reshape(1, 227, 227, 3).transpose(0, 3, 1, 2).astype(np.float32)
        #     model.predictor(img, data=i)
        # - - -
        for index, x in enumerate(x_data[970:1000]):
            # for i in range(5):
                cnt += 1
                print((x.reshape((32, 32)) * 255).astype(int).shape)
                x = lgpy.blur((x.reshape((32, 32)) * 255).astype(int), 1, 32)
                # ximg = cv2.resize(ximg, (32, 32))
                # ximg = cv2.cvtColor(ximg, cv2.COLOR_RGB2GRAY)
                # img = ximg.astype(np.float32) / 255.0
                img = x.reshape((1, 32 * 32)).astype(np.float32) / 255
                # img = img.transpose((2, 0, 1)).reshape((1, 3, 227, 227))
                p = model.predictor(img)
                p = F.softmax(p)
                tg = np.argmax(p.data)
                print(tg, labelname[tg])
                # pylab.savefig("./result_{}.png".format(index))
                pylab.subplot(10, 6, cnt)
                pylab.axis("off")
                pylab.imshow(x.reshape((32, 32)))
                cnt += 1
                pylab.subplot(10, 6, cnt)
                pylab.axis("off")
                ok = Image.open("../serebii/{}.png".format(labelname[tg]))
                pylab.imshow(ok)
    pylab.show()
    # pylab.savefig("/Users/ctare/Desktop/result.png")
    # x_data = np.load("sixdata227.npy")
    #
    # #  - - lea im - -
    # x_tests = np.load("lg/xtdata/x_poketestLG.npy")
    # #
    # # x_gyara = x_tests[1 * 10][:, :, ::-1]
    # # x_gyara = np.array([cv2.resize(x_gyara, (227, 227))], dtype=np.float32).transpose(0, 3, 1, 2)
    # #
    # # x_kaguya = x_tests[0 * 10][:, :, ::-1]
    # # x_kaguya = np.array([cv2.resize(x_kaguya, (227, 227))], dtype=np.float32).transpose(0, 3, 1, 2)
    # #
    # # x_pori = x_tests[5 * 10][:, :, ::-1]
    # # x_pori = np.array([cv2.resize(x_pori, (227, 227))], dtype=np.float32).transpose(0, 3, 1, 2)
    # #
    # # x_gard = x_tests[2 * 10][:, :, ::-1]
    # # x_gard = np.array([cv2.resize(x_gard, (227, 227))], dtype=np.float32).transpose(0, 3, 1, 2)
    # # imodel(x_gyara)
    # # cnt += 1
    # # imodel(x_kaguya)
    # # cnt += 1
    # # imodel(x_pori)
    # # cnt += 1
    # # imodel(x_gard)
    # # pylab.show()
    # # - - - - - 
    #
    # for i in range(6):
    #     # x_test = np.r_[np.zeros((25, 100, 3), dtype=np.uint8), x_test[:-25, :, ::-1]]
    #     for idx, x_test in enumerate(x_tests[i*10: i*10 + 10]):
    #         x_test = x_test[:, :, :]
    #         x_tmp = x_test
    #
    #         x_test = np.array([cv2.resize(x_test, (227, 227))], dtype=np.float32).transpose(0, 3, 1, 2)
    #
    #         # x_data = np.load("sixdata227.npy").astype(np.float32) / 255
    #         # t_data = np.load("sixlabel.npy").astype(np.int32)
    #         #
    #         # x_data = x_data.transpose(0, 3, 1, 2)
    #         #
    #         # print(x_data.shape)
    #         p = model.predictor(x_test)
    #         # print(p)
    #         # print("---")
    #         # print(p.shape)
    #         # print("===")
    #         p = F.softmax(p)
    #         # print(p)
    #         print(np.argmax(p.data))
    #
    #         pylab.subplot(14, 10, idx + 20*i + 1 )
    #         pylab.axis('off')
    #         pylab.imshow(x_tmp.astype(np.uint8))
    #         pylab.title("%d" % np.argmax(p.data))
    # for s in range(6):
    #     pylab.subplot(14, 10, 120 + s + 1)
    #     pylab.axis('off')
    #     pylab.imshow(x_data[200 * s + 1])
    #     pylab.title(s)
    # print("===")
    # pylab.show()
    # print(x_test.shape)
    # print(sum(np.argmax(p.data, axis=1) == t_test) / len(t_test))
else:
    x_train = chainer.datasets.TupleDataset(x_data, t_data)
    train_itr = chainer.iterators.SerialIterator(x_train, batch_size=320)
    print(x_data.shape)
    print(t_data.shape)

    try:
        import cupy
        if cupy.available:
            updater = chainer.training.StandardUpdater(train_itr, optimizer, device=device)
        else:
            updater = chainer.training.StandardUpdater(train_itr, optimizer)
    except:
        updater = chainer.training.StandardUpdater(train_itr, optimizer)
    trainer = chainer.training.Trainer(updater, (5, "epoch"))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(["epoch", "main/loss", "main/accuracy"]))
    trainer.extend(extensions.ProgressBar())

    print("run!")
    trainer.run()
    chainer.serializers.save_npz(model_path, model)

# c1 = L.Convolution2D(3, 96, 12, stride=4)
# h = c1(x_data, dtype=np.float32))
# print(h.shape)
# h = F.max_pooling_2d(h, 3, stride=2)
# print(h.shape)
# c2 = L.Convolution2D(96, 256, 5, pad=2)
# h = c2(h)
# print(h.shape)
# h = F.max_pooling_2d(h, 3, stride=2)
# print(h.shape)
# c3 = L.Convolution2D(256, 384, 3, pad=1)
# h = c3(h)
# print(h.shape)
# c4 = L.Convolution2D(384, 384, 3, pad=1)
# h = c4(h)
# print(h.shape)
# c5 = L.Convolution2D(384, 256, 3, pad=1)
# h = c5(h)
# print(h.shape)
# h = F.max_pooling_2d(h, 3, stride=2)
# print(h.shape)
# l1 = L.Linear(256 * 6 * 6, 4096)
# h = l1(h)
# print(h.shape)
# l2 = L.Linear(4096, 1024)
# h = l2(h)
# print(h.shape)
# l3 = L.Linear(1024, 6)
# h = l3(h)
# print(h.shape)
# print(F.softmax(h))
