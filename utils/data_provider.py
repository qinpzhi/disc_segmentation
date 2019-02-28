# -*- coding: utf-8 -*-
'''
@Time    : 18-9-5 上午11:26
@Author  : qinpengzhi
@File    : data_provider.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import cv2
from PIL import Image
import os
import glob
import numpy as np
from networks.config import data_cfg

class DataProvider(object):
    def __init__(self,config):
        self.cfg=config
        self.file_idx=-1
        self.label_train,self.label_mask=self._find_label_files(config)
        # print self.label_train
        # print self.label_mask

    #获取的是train和mask的全部路径
    # def _find_label_files(self,config):
    #     label_train=[]
    #     label_mask=[]
    #     for item in config.item_list:
    #         label_train.extend(sorted(glob.glob(os.path.join(config.train_path,item,"*jpg"))))
    #         label_mask.extend(sorted(glob.glob(os.path.join(config.mask_path,item,"*bmp"))))
    #     return label_train,label_mask
    def _find_label_files(self,config):
        label_train=[]
        label_mask=[]
        label_train.extend(sorted(glob.glob(os.path.join(config.train_path,"*jpg"))))
        label_mask.extend(sorted(glob.glob(os.path.join(config.mask_path,"*bmp"))))
        label_train=np.array(label_train)
        label_mask=np.array(label_mask)
        return label_train,label_mask

    def next_batch(self,batch_size=1):
        self.cycle_files()
        im_data=self.load_image_data(self.label_train[self.file_idx])
        im_mask=self.load_image_mask(self.label_mask[self.file_idx])
        im_data=im_data[np.newaxis,:]
        im_data=im_data[:,:,:,1:]
        im_mask = im_mask[np.newaxis,:,:,np.newaxis]
        # print np.shape(im_data)
        # print np.shape(im_mask)
        return im_data,im_mask

    def load_image_data(self,data_file,dtype=np.float32):
        try:
            tempimage=Image.open(data_file)
            # tempimage=tempimage.resize((512,512))
            return np.array(tempimage,dtype)
            #img = Image.fromarray(img.astype(dtype)).convert('RGB')
        except Exception, e:
            print ('#######')
            print (data_file + '\n' + e.message)
            return None
    def load_image_mask(self,data_file,dtype=np.int32):
        try:
            # print data_file
            tempimage = Image.open(data_file)
            # tempimage = tempimage.resize((512, 512))
            image_mask=np.array(tempimage,dtype)
            image_mask=image_mask[:,:,1]
            for i in range(np.shape(image_mask)[0]):
                for j in range(np.shape(image_mask)[1]):
                    if image_mask[i][j]==255:
                        image_mask[i][j]=0
                    elif image_mask[i][j]==128:
                        image_mask[i][j]=2
                    elif image_mask[i][j]==0:
                        image_mask[i][j]=1
            return image_mask
            #img = Image.fromarray(img.astype(dtype)).convert('RGB')
        except Exception, e:
            print ('#######')
            print (data_file + '\n' + e.message)
            return None

    def cycle_files(self):
        self.file_idx+=1
        if self.file_idx>=len(self.label_train):
            ##打乱顺序
            perm = np.arange(len(self.label_train))
            np.random.shuffle(perm)
            self.label_train = self.label_train[perm]
            self.label_mask = self.label_mask[perm]
            # print self.label_train[0], self.label_mask[0]
            self.file_idx=0

# import cv2 as cv
# img=np.array(cv.imread("/home/qpz/data/REFUGE/preprocess_data/data/g0001_1.jpg"))
# print np.shape(img)
# a=set([])
# for i in range(np.shape(img)[0]):
#     for j in range(np.shape(img)[1]):
#         # print i," ",j," ",img[i][j][0]," ",img[i][j][1]," ",img[i][j][2]
#         # print
#
#         a.add(img[i][j][2])
# print a
# a=DataProvider(data_cfg)
# a.next_batch()