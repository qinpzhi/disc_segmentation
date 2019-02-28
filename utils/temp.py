# -*- coding: utf-8 -*-
'''
@Time    : 18-9-17 下午3:25
@Author  : qinpengzhi
@File    : temp.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import cv2
import numpy as np
origa = cv2.imread('/home/qpz/data/followupImg/followupImg/1710/left/11034910.jpg')
# origa=cv2.resize(origa,(512,512))
img=cv2.cvtColor(origa,cv2.COLOR_BGR2GRAY)

img = cv2.medianBlur(img,5)
# print np.shape(img)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,40,param1=39,param2=26,minRadius=5,maxRadius=350)
# print circles
circles = np.uint16(np.around(circles))
circles = circles[0][0]
cv2.circle(origa,(circles[0],circles[1]),circles[2],(0,255,0),2)
cv2.rectangle(origa,(circles[0]-64,circles[1]-64),(circles[0]+64,circles[1]+64),(255,0,0),2)
# for i in circles[0,:]:
#     cv2.circle(origa,(i[0],i[1]),i[2],(0,255,0),2)
#     cv2.circle(origa, (i[0], i[1]), 2, (0, 255, 0), 2)
cv2.imshow('detected circles',img)
cv2.imshow('imgtt',origa)
cv2.waitKey(0)
cv2.destroyAllWindows()