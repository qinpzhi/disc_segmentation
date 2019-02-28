# -*- coding: utf-8 -*-
'''
@Time    : 18-9-5 下午6:41
@Author  : qinpengzhi
@File    : Segmentation_train.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import _init_pathes
from networks.config import data_cfg,train_cfg,model_cfg
from pprint import pprint
import os
import tensorflow as tf
from utils.data_provider import DataProvider
from utils.logger import TfLogger
from models.seg_model import SegModel
from trainers.seg_trainer import SegTrainer

def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
def main():
    print('---------------------- data config ------------------------')
    pprint(data_cfg)

    print('---------------------- model config -------------------')
    pprint(model_cfg)

    print('creating dirs for saving model weights, logs ...')
    checkpoint_dir = os.path.join(
        model_cfg.checkpoint_dir, model_cfg.exp_name)
    create_dirs([checkpoint_dir, train_cfg.summary_dir])

    print('initializing train data provider....')
    det_data_provider = DataProvider(data_cfg)

    sess = tf.Session()
    print('creating tensorflow log for summaries...')
    tf_logger = TfLogger(sess, train_cfg)
    print('creating seg models ...')
    train_model = SegModel(model_cfg)
    if model_cfg.train_from_pretrained:
        train_model.load(sess)

    print('creating seg trainer...')
    trainer = SegTrainer(sess, train_model,
                         det_data_provider, train_cfg, tf_logger)

    print('start trainning...')
    trainer.train()

    sess.close()


if __name__ == '__main__':
    main()
