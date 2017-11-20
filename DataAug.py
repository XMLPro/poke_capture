# coding: utf-8

# 画像拡張の関数

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 画像のグレースケール化(雑)
def gray_image(img):
    meanimg = np.mean(img, axis=2)
    grayimg = np.zeros_like(img)
    for i in range(3):
        grayimg[:,:,i] = meanimg
    return grayimg


# 輝度(彩度)変更
def brightness_image(img):
    dst = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            # Vの値を上書きする
            dst[i, j, 1] = int(dst[i, j, 1] * 0.5)
    img2 = cv2.cvtColor(dst, cv2.COLOR_HSV2RGB)  # HSVからBGRに戻す
    return img2

# 傾きの変換(アフィン変換)
def affine_image(img):
    rows,cols,ch = img.shape

    pts1 = np.float32([[20,20],[80,20],[50,100]])
    pts2 = np.float32([[10,10],[70,10],[50,100]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

# 画像縮小(アフィン変換)
def small_image(img):    
    rows,cols,ch = img.shape

    pts1 = np.float32([[20,20],[80,20],[50,100]])
    pts2 = np.float32([[30,30],[70,30],[50,80]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst


# 画像の一部をマスク
def mask_image(img):
    maskedimg = np.copy(img)
    maskedimg[60:80,70:90,:] = np.array([255,255,0])
    return maskedimg

# 画像の反転
def mirror_image(img):
    mirrorW = img[:,::-1,:]
    mirrorH = img[::-1,:,:]
    return mirrorW, mirrorH