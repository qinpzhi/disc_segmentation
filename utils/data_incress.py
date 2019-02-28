# -*- coding: utf-8 -*-
'''
@Time    : 18-9-19 下午2:50
@Author  : qinpengzhi
@File    : data_incress.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import numpy as np
import random
import skimage
import io,os
import glob
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance
import cv2
from networks.config import data_cfg
#root_path为图像根目录，img_name为图像名字
root_path="/home/qpz/data/temp"

# def move(path,off): #平移，平移尺度为off
#     img = Image.open(path)
#     offset = ImageChops.offset(off,0)
#     return offset

def flip(path):   #翻转图像
    img = Image.open(path)
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

def aj_contrast(path): #调整对比度 两种方式 gamma/log
    image = scipy.misc.imread(path)
    gam= skimage.exposure.adjust_gamma(image, 0.8)
    # skimage.io.imsave(os.path.join(root_path,img_name.split('.')[0] + '_gam.jpg'),gam)
    log= skimage.exposure.adjust_log(image)
    # skimage.io.imsave(os.path.join(root_path,img_name.split('.')[0] + '_log.jpg'),log)
    return gam,log
def rotation(path,rot):
    img = Image.open(path)
    rotation_img = img.rotate(rot) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def randomGaussian(root_path, img_name, mean, sigma):  #高斯噪声
    image = Image.open(os.path.join(root_path, img_name))
    im = np.array(image)
    #设定高斯函数的偏移
    means = 0
    #设定高斯函数的标准差
    sigma = 25
    #r通道
    r = im[:,:,0].flatten()

    #g通道
    g = im[:,:,1].flatten()

    #b通道
    b = im[:,:,2].flatten()

    #计算新的像素值
    for i in range(im.shape[0]*im.shape[1]):

        pr = int(r[i]) + random.gauss(0,sigma)

        pg = int(g[i]) + random.gauss(0,sigma)

        pb = int(b[i]) + random.gauss(0,sigma)

        if(pr < 0):
            pr = 0
        if(pr > 255):
            pr = 255
        if(pg < 0):
            pg = 0
        if(pg > 255):
            pg = 255
        if(pb < 0):
            pb = 0
        if(pb > 255):
            pb = 255
        r[i] = pr
        g[i] = pg
        b[i] = pb
    im[:,:,0] = r.reshape([im.shape[0],im.shape[1]])

    im[:,:,1] = g.reshape([im.shape[0],im.shape[1]])

    im[:,:,2] = b.reshape([im.shape[0],im.shape[1]])
    gaussian_image = Image.fromarray(np.uint8(im))
    return gaussian_image
def randomColor(path): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(path)
    random_factor = np.random.randint(0, 10) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(4, 10) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(4, 11) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 10) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度



def find_label_files(config):
    label_train = []
    label_mask = []
    label_train.extend(sorted(glob.glob(os.path.join(config.train_path, "*jpg"))))
    label_mask.extend(sorted(glob.glob(os.path.join(config.mask_path, "*bmp"))))
    return label_train, label_mask

label_train, label_mask=find_label_files(data_cfg)

for i in range(len(label_train)):
    mask_path=label_mask[i]
    train_path=label_train[i]
    img_name = train_path.split('/')[-1].split('.')[0]
    iii=Image.open(train_path)
    iii.save("/home/qpz/data/data_incress/data/"+img_name+"_0.jpg")
    jjj=Image.open(mask_path)
    jjj.save("/home/qpz/data/data_incress/mask/" + img_name + "_0.bmp")

    scipy.misc.imsave("/home/qpz/data/data_incress/data/"+img_name+"_1.jpg",aj_contrast(train_path)[0])
    Image.open(mask_path).save("/home/qpz/data/data_incress/mask/" + img_name + "_1.bmp")

    flp1=flip(train_path)
    flp1.save("/home/qpz/data/data_incress/data/"+img_name+"_2.jpg")
    fm1=flip(mask_path)
    fm1.save("/home/qpz/data/data_incress/mask/" + img_name + "_2.bmp")

    rotation(train_path, 90).save("/home/qpz/data/data_incress/data/"+img_name+"_3.jpg")
    rotation(mask_path, 90).save("/home/qpz/data/data_incress/mask/" + img_name + "_3.bmp")

    rotation(train_path, 180).save("/home/qpz/data/data_incress/data/"+img_name+"_4.jpg")
    rotation(mask_path, 180).save("/home/qpz/data/data_incress/mask/" + img_name + "_4.bmp")

    rotation(train_path,270).save("/home/qpz/data/data_incress/data/"+img_name+"_5.jpg")
    rotation(mask_path, 270).save("/home/qpz/data/data_incress/mask/" + img_name + "_5.bmp")




# for img_path in label_train:
#     img_name=img_path.split('/')[-1].split('.')[0]
#     orig1 = cv.imread(img_path)
#     b, g, r = cv.split(orig1)
#     zeros = np.zeros(orig1.shape[:2], dtype="uint8")
#     merged1 = cv.merge([b, g, zeros])
#     merged2 = cv.merge([b, g, r])
#
#     orig2 = cv.imread(img_path)
#     orig2 = cv.flip(orig2,1)
#     b, g, r = cv.split(orig2)
#     zeros = np.zeros(orig2.shape[:2], dtype="uint8")
#     merged3 = cv.merge([b, g, zeros])
#     merged4 = cv.merge([b, g, r])
#     cv.imwrite("/home/qpz/data/REFUGE/preprocess_data/data/"+img_name+"_1.jpg",merged1)
#     cv.imwrite("/home/qpz/data/REFUGE/preprocess_data/data/" + img_name + "_2.jpg", merged3)
# for img_path in label_mask:
#     img_name=img_path.split('/')[-1].split('.')[0]
#     orig1 = cv.imread(img_path)
#
#     orig2 = cv.flip(orig1,1)
#
#     cv.imwrite("/home/qpz/data/REFUGE/preprocess_data/mask/"+img_name+"_1.bmp",orig

