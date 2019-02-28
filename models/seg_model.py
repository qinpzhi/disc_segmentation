# -*- coding: utf-8 -*-
'''
@Time    : 18-9-5 下午2:19
@Author  : qinpengzhi
@File    : seg_model.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
from networks.config import model_cfg
from base.base_model import BaseModel
import utils.TensorflowUtils as utils
import  numpy as np

class SegModel(BaseModel):
    def __init__(self, config):
        super(SegModel, self).__init__(config)
        self._scope = 'segmodel'
        self._nclass=config.number_of_classes
        self.learning_rate=config.learning_rate ##Learning rate for Adam Optimizer
        self.build_model()
        self.init_saver()

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.saver_max_to_keep)

    def build_model(self):
        # tf.reset_default_graph()
        self.data = tf.placeholder(tf.float32, shape=[1, 512, 512, 2])
        self.mask = tf.placeholder(tf.int32, shape=[None, None, None, 1])
        #used for dropout to alleviate over-fitting issue
        self.keep_prob = tf.placeholder(tf.float32)
        with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.fully_connected],\
                            weights_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01),\
                            weights_regularizer = tf.contrib.layers.l2_regularizer(self.config.weight_decay),\
                            biases_initializer = tf.constant_initializer(0.0)):
            with tf.variable_scope(self._scope):
                conv1 = slim.repeat(self.data, 2, slim.conv2d, 64, [3, 3], padding='SAME', scope='conv1')
                print "conv1:" ,conv1.get_shape()
                pool1 = slim.max_pool2d(conv1, [2, 2], padding='VALID', scope='pool1')
                print "pool1:", pool1.get_shape()
                conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], padding='SAME', scope='conv2')  # stride=2
                pool2 = slim.max_pool2d(conv2, [2, 2], padding='VALID', scope='pool2')
                print "pool2:", pool2.get_shape()
                conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3], padding='SAME', scope='conv3')  # stride=4
                pool3 = slim.max_pool2d(conv3, [2, 2], padding='VALID', scope='pool3')
                print "pool3:", pool3.get_shape()
                conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], padding='SAME', scope='conv4')  # stride=8
                pool4 = slim.max_pool2d(conv4, [2, 2], padding='VALID', scope='pool4')
                conv5 = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], padding='SAME', scope='conv5')  # stride=16
                print "conv5:", conv5.get_shape()
                pool5 = slim.max_pool2d(conv5, [2, 2], padding='VALID', scope='pool5')
                conv6 = slim.conv2d(pool5, 4096, [7, 7], padding="SAME")
                print "conv6:", conv6.get_shape()
                relu6 = tf.nn.relu(conv6, name="relu6")
                relu_dropout6 = tf.nn.dropout(relu6, keep_prob=self.keep_prob)
                conv7 = slim.conv2d(relu_dropout6, 4096, [1, 1], padding="SAME")
                relu7 = tf.nn.relu(conv7, name="relu7")
                relu_dropout7 = tf.nn.dropout(relu7, keep_prob=self.keep_prob)
                conv8 = slim.conv2d(relu_dropout7, self._nclass, [1, 1], padding="SAME")
                print "pool4:", pool4.get_shape()
                print "conv8:", conv8.get_shape()
                # conv_t1 = slim.conv2d_transpose(conv8, pool4.get_shape()[3].value,[4, 4], stride=2, padding="SAME")
                # conv_t1 = tf.nn.conv2d_transpose()

                deconv_shape1 = pool4.get_shape()
                W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, self._nclass], name="W_t1")
                b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
                conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(pool4))

                print "conv_t1:", conv_t1.get_shape()
                print "pool4:", pool4.get_shape()
                fuse_1 = tf.add(conv_t1, pool4, name="fuse_1")
                print "fuse_1:", fuse_1.get_shape()

                conv_add1 = slim.conv2d(fuse_1, 256, [1, 1], padding="SAME",scope="conv_add1")
                conv_add2 = slim.conv2d(conv_add1, 256, [3, 3], padding="SAME",scope="conv_add2")
                conv_add3 = slim.conv2d(conv_add2, 512, [1, 1], padding="SAME",scope="conv_add3")

                deconv_shape2 = pool3.get_shape()
                W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
                b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
                conv_t2 = utils.conv2d_transpose_strided(conv_add3, W_t2, b_t2, output_shape=tf.shape(pool3))
                # conv_t2 = slim.conv2d_transpose(fuse_1, pool3.get_shape()[3].value, [4, 4], stride=2, padding="SAME")
                print "conv_t2:", conv_t2.get_shape()
                fuse_2 = tf.add(conv_t2, pool3, name="fuse_2")
                print "fuse_2:", fuse_2.get_shape()

                conv_add4 = slim.conv2d(fuse_2, 128, [1, 1], padding="SAME",scope="conv_add4")
                conv_add5 = slim.conv2d(conv_add4, 128, [3, 3], padding="SAME",scope="conv_add5")
                conv_add6 = slim.conv2d(conv_add5, 256, [1, 1], padding="SAME",scope="conv_add6")

                shape = tf.shape(self.data)
                deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self._nclass])
                W_t3 = utils.weight_variable([16, 16, self._nclass, deconv_shape2[3].value], name="W_t3")
                b_t3 = utils.bias_variable([self._nclass], name="b_t3")
                conv_t3 = utils.conv2d_transpose_strided(conv_add6, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
                print "conv_t3:", conv_t3.get_shape()
                # conv_t3 = slim.conv2d_transpose(fuse_2, self._nclass, [16, 16], stride=8)





                self.logits=conv_t3
                self.annotation_pred = tf.argmax(conv_t3, axis=3, name="prediction")
                self._build_loss()

    def _build_loss(self):
        with tf.variable_scope('loss') as scope:
            self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                               labels=tf.squeeze(self.mask,axis=[3]),name="entropy")))

            self.pixelAcc =tf.reduce_sum(tf.cast(tf.equal(self.annotation_pred, tf.cast(tf.squeeze(self.mask, axis=[3]), tf.int64)), tf.float32)) / tf.cast(512*512, tf.float32)
            self.pixelIOU=self._getIOU()


            # self.lr = tf.train.exponential_decay(self.config.learning_rate, self.global_step_tensor, \
            #                                      self.config.step_size, 0.1, staircase=True)
          #   optimizert = tf.train.AdamOptimizer(self.learning_rate)
          #   trainable_var = tf.trainable_variables()
          #   grads = optimizert.compute_gradients(self.loss, var_list=trainable_var)
          #   self.optimizer = optimizert.apply_gradients(grads)
          #   self.optimizer = tf.train.AdamOptimizer(learning_rate = self.config.learning_rate).minimize(self.loss)

            # self.optimizer = tf.train.MomentumOptimizer(self.lr, self.config.momentum).minimize(self.loss, \
            #                                                                                     global_step=self.global_step_tensor)
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, \
                                                                                                     global_step=self.global_step_tensor)
    def _getIOU(self):
        len1=tf.shape(tf.where(self.annotation_pred>0))[0]
        len2=tf.shape(tf.where(tf.squeeze(self.mask, axis=[3])>0))[0]
        temppred=self.annotation_pred
        tempact=tf.cast(tf.squeeze(self.mask, axis=[3]), tf.int64)

        temppred=tf.where(tf.equal(temppred,0),temppred-1,temppred)

        len3=tf.reduce_sum(
            tf.cast(tf.equal(temppred, tempact),
                    tf.int32))
        pixelIOU = tf.reduce_sum(
            tf.cast(tf.equal(temppred, tempact),
                    tf.float32)) / tf.cast(len1+len2-len3, tf.float32)
        return pixelIOU

# aaa=SegModel(model_cfg)