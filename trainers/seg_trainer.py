# -*- coding: utf-8 -*-
'''
@Time    : 18-9-5 下午5:02
@Author  : qinpengzhi
@File    : seg_trainer.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

class SegTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(SegTrainer,self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop=tqdm(range(self.config.num_iter_per_epoch))
        seg_losses=[]
        seg_pixelAcc=[]
        seg_pixelIOU=[]
        for it in loop:
            loss,pixelAcc,pixelIOU=self.train_step()
            seg_losses.append(loss)
            seg_pixelAcc.append(pixelAcc)
            seg_pixelIOU.append(pixelIOU)
            print pixelAcc
            print pixelIOU

        agv_seg_loss=np.mean(seg_losses)
        agv_seg_pixelAcc=np.mean(seg_pixelAcc)
        agv_seg_pixelIOU=np.mean(pixelIOU)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        print 'seg_losses=%f;seg_pixelAcc=%f;seg_pixelIOU=%f;itr=%d' % (agv_seg_loss,agv_seg_pixelAcc,agv_seg_pixelIOU,cur_it)

        summaries_dict = {}
        summaries_dict['seg_losses'] = agv_seg_loss
        summaries_dict['seg_pixelAcc'] = agv_seg_pixelAcc
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        self.model.save(self.sess)

    def train_step(self):
        im_data, im_mask=self.data.next_batch()
        feed_dict={self.model.data:im_data, self.model.mask:im_mask,self.model.keep_prob:0.85}
        seg_loss,pixelAcc,pixelIOU, _ =self.sess.run([self.model.loss, self.model.pixelAcc,self.model.pixelIOU,self.model.optimizer],feed_dict = feed_dict)
        return seg_loss,pixelAcc,pixelIOU
