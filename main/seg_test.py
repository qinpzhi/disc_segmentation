# -*- coding: utf-8 -*-
'''
@Time    : 18-9-6 下午7:16
@Author  : qinpengzhi
@File    : seg_test.py
# @Software: PyCharm
@Contact : qinpzhi@163.com
'''
import _init_pathes
import tensorflow as tf
from models.seg_model import SegModel
from networks.config import data_cfg,model_cfg,train_cfg
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as pyplot

if __name__=='__main__':
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    det_test_model = SegModel(model_cfg)
    det_test_model.load(sess)
    im_path = '/media/qpz/xuschang/zj_data/qqqq/*.jpg'
    im_files = glob.glob(im_path)
    # self.annotation_pred
    for im_file in im_files:
        print 'start' +im_file
        im = Image.open(im_file)
        im_feed = np.array(im.copy(), dtype=np.float32)
        im_feed = im_feed[np.newaxis, :]
        #im_feed = np.resize(im_feed, (1, 1024, 1024, 3))
        pred = sess.run(det_test_model.annotation_pred, feed_dict={det_test_model.data: im_feed,
                                                    det_test_model.keep_prob: 1.0})
        pred =tf.squeeze(pred)
        pred=pred.eval(session=sess)
        aaa=set([])
        print type(pred)
        for i in range(np.shape(pred)[0]):
            for j in range(np.shape(pred)[1]):
                aaa.add(pred[i][j])
                if pred[i][j] == 1:
                    pred[i][j] = 255
                elif pred[i][j] == 2:
                    pred[i][j] = 128
        print aaa

        pyplot.imshow(pred)
        pyplot.show()

