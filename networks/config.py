# -*- coding: utf-8 -*-
'''
@Time    : 18-9-5 上午11:06
@Author  : qinpengzhi
@File    : config.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import os
import os.path as osp
from easydict import EasyDict as edict

#设置数据相关的config
data_cfg=edict()
data_cfg.train_path="/home/qpz/data/REFUGE/preprocess_new1/data"
data_cfg.mask_path="/home/qpz/data/REFUGE/preprocess_new1/mask"
data_cfg.item_list=["Glaucoma","Non-Glaucoma"]
#设置model相关参数
model_cfg=edict()
model_cfg.exp_name = 'seg_model'
model_cfg.checkpoint_dir = '../checkpoints'
model_cfg.train_from_pretrained=False
model_cfg.saver_max_to_keep = 2
model_cfg.weight_decay = 0.0004
model_cfg.number_of_classes=3
model_cfg.learning_rate=1e-4 ##Learning rate for Adam Optimizer
model_cfg.momentum = 0.9
model_cfg.step_size = 500
#设置train相关参数
train_cfg=edict()
train_cfg.summary_dir = '../summaries/seg'
train_cfg.num_iter_per_epoch=100
train_cfg.num_epochs=1000

