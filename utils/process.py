# -*- coding: utf-8 -*-
'''
@Time    : 18-9-12 上午9:29
@Author  : qinpengzhi
@File    : process.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import cv2
import numpy as np
import os
import glob

import cv2
import numpy as np
# from matplotlib import pyplot as plt
# img=cv2.imread('/home/qpz/data/REFUGE/preprocess_new/data/g0022.jpg')
# #cv2.CV_64F 输出图像的深度(数据类型),可以使用 -1, 与原图像保持一致 np.uint8
# kernel = np.array([[ 0, -1, 0],[ -1, 5.5, -1,],[0, -1, 0,]])
# dst = cv2.filter2D(img,-1,kernel)
# # b, g, r = cv2.split(dst)
# # bmax=np.max(b)
# # gmax=np.max(g)
# # rmax=np.max(r)
# # bmax=np.max(b)
# # gmax=np.max(g)
# # rmax=np.max(r)
# #
# # b=b/bmax*255
# #
# # g=g/gmax*255
# #
# # r=r/rmax*255
# # zeros = np.zeros(dst.shape[:2], dtype = "uint8")
# # merged4 = cv2.merge([b, g, zeros])
#
# cv2.imshow('image',img)
# cv2.imshow('dst',dst)
# # cv2.imshow('merged4',merged4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# def find_label_files():
#     label_train = []
#     label_mask = []
#     label_train.extend(sorted(glob.glob(os.path.join("/home/qpz/data/REFUGE/origa_new/Training400/", "*jpg"))))
#     label_mask.extend(sorted(glob.glob(os.path.join("/home/qpz/data/REFUGE/origa_new/Disc_Cup_Masks/", "*bmp"))))
#     return label_train, label_mask
#
# label_train, label_mask=find_label_files()
# print len(label_mask)
# for i in range(len(label_mask)):
#     img_name0 = label_mask[i].split('/')[-1].split('.')[0]
#     img_name1 = label_train[i].split('/')[-1].split('.')[0]
#
#     # print img_name0,img_name1
#     img = cv2.imread(label_mask[i], 0)
#     img1 = cv2.imread(label_train[i])
#     img = cv2.medianBlur(img, 5)
#     cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 40,
#                                param1=10, param2=11, minRadius=75, maxRadius=500)
#     if circles is None:
#         print img_name0, img_name1
#         continue
#     circles = np.uint16(np.around(circles))
#     circles = circles[0][0]
#
#     circles[1] = max(256, circles[1])
#     circles[0] = max(256, circles[0])
#     circles[1] = min(circles[1], 1800)
#     circles[0] = min(circles[0], 1868)
#     origaimg = cimg[circles[1] - 256:circles[1] + 256, circles[0] - 256:circles[0] + 256]
#     origaimg1 = img1[circles[1] - 256:circles[1] + 256, circles[0] - 256:circles[0] + 256]
#
#     kernel = np.array([[0, -1, 0], [-1, 5.5, -1, ], [0, -1, 0, ]])
#     origaimg1 = cv2.filter2D(origaimg1, -1, kernel)
#     cv2.imwrite("/home/qpz/data/REFUGE/preprocess_new1/mask/"+img_name0+".bmp",origaimg)
#     cv2.imwrite("/home/qpz/data/REFUGE/preprocess_new1/data/" + img_name1 + ".jpg", origaimg1)
def find_label_files():
    label_train = []
    label_mask = []
    label_train.extend(sorted(glob.glob(os.path.join("/home/qpz/data/REFUGE/origa_new/Training400/", "*jpg"))))
    return label_train

label_train=find_label_files()
print len(label_train)
for i in range(len(label_train)):
    img_name0 = label_train[i].split('/')[-1].split('.')[0]

    # print img_name0,img_name1
    img = cv2.imread(label_train[i], 0)
    # img1 = cv2.imread(label_train[i])
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 40,
                               param1=10, param2=11, minRadius=75, maxRadius=500)
    if circles is None:
        print img_name1
        continue
    circles = np.uint16(np.around(circles))
    circles = circles[0][0]

    circles[1] = max(256, circles[1])
    circles[0] = max(256, circles[0])
    circles[1] = min(circles[1], 1800)
    circles[0] = min(circles[0], 1868)
    origaimg = cimg[circles[1] - 256:circles[1] + 256, circles[0] - 256:circles[0] + 256]
    # origaimg1 = img1[circles[1] - 256:circles[1] + 256, circles[0] - 256:circles[0] + 256]

    kernel = np.array([[0, -1, 0], [-1, 5.5, -1, ], [0, -1, 0, ]])
    cv2.imshow()
    # origaimg1 = cv2.filter2D(origaimg1, -1, kernel)
    cv2.imwrite("/home/qpz/data/REFUGE/preprocess_new1/mask/"+img_name0+".bmp",origaimg)
    # cv2.imwrite("/home/qpz/data/REFUGE/preprocess_new1/data/" + img_name1 + ".jpg", origaimg1)



