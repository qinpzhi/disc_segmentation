# -*- coding: utf-8 -*-
'''
@Time    : 18-9-6 下午1:52
@Author  : qinpengzhi
@File    : img_Preprocessing.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
##对图片进行预处理
import numpy as np
from PIL import Image
# import ImageFilter
# # img=Image.open("/home/qpz/data/REFUGE/Disc_Cup_Masks/Disc_Cup_Masks/Glaucoma/g0011.bmp")
# img=Image.open("/home/qpz/data/REFUGE/REFUGE-Training400/Training400/Glaucoma/g0001.jpg")
#
# img=img.resize((512, 512))
# img1=img.transpose(Image.FLIP_LEFT_RIGHT)
# img.show()
# img1.show()
# imfilter = img.filter(ImageFilter.DETAIL)

import cv2 as cv
import os
import glob
from networks.config import data_cfg

# def find_label_files(config):
#     label_train = []
#     label_mask = []
#     for item in config.item_list:
#         label_train.extend(sorted(glob.glob(os.path.join(config.train_path, item, "*jpg"))))
#         label_mask.extend(sorted(glob.glob(os.path.join(config.mask_path, item, "*bmp"))))
#     return label_train, label_mask


orig = cv.imread("/home/qpz/data/REFUGE/REFUGE-Training400/Training400/Glaucoma/g0001.jpg")
orig=cv.resize(orig,(512,512))
orig1 = cv.imread("/home/qpz/data/REFUGE/Disc_Cup_Masks/Disc_Cup_Masks/Glaucoma/g0001.bmp")
orig1=cv.resize(orig1,(512,512))
orig2 = cv.flip(orig1,1)
orig2 = cv.flip(orig,0)
b, g, r = cv.split(orig)
print np.shape(r)
ass=set([])
for i in range(np.shape(r)[0]):
    for j in range(np.shape(r)[1]):
        ass.add(r[i][j])
print ass
cv.imshow("Blue 1", b)
cv.imshow("Green 1", g)
cv.imshow("Red 1", r)
cv.imshow("mask", orig1)
zeros = np.zeros(orig.shape[:2], dtype = "uint8")
merged1=cv.merge([b,g,zeros])
# merged2=cv.merge([b,g,r])
# /home/qpz/data/REFUGE/REFUGE-Training400/Training400/Glaucoma
cv.imwrite("/home/qpz/data/REFUGE/preprocess_data/data/dd.jpg", merged1)
cv.waitKey(0)
cv.destoryAllWindows()

# label_train, label_mask=find_label_files(data_cfg)
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
#     cv.imwrite("/home/qpz/data/REFUGE/preprocess_data/mask/"+img_name+"_1.bmp",orig1)
#     cv.imwrite("/home/qpz/data/REFUGE/preprocess_data/mask/" + img_name + "_2.bmp", orig2)

# tempimage = Image.open("aaa.png")
#             # tempimage = tempimage.resize((512, 512))
# image_mask=np.array(tempimage,np.int32)
# print np.shape(image_mask)
# # image_mask=image_mask[:,:,1]
# aaa=set([])
# for i in range(np.shape(image_mask)[0]):
#     for j in range(np.shape(image_mask)[1]):
#         aaa.add(image_mask[i][j])
# print aaa