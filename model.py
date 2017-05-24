#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator

class Model(object):
    def __init__(self, z_dim, batch_size, clip_threshold):

        self.input_size = 256
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.clip_threshold = clip_threshold
        
        # generator config
        gen_conv_layer = [3, 64, 128, 256, 512, 512, 512, 512, 512]
        gen_deconv_layer = [512, 512, 512, 512, 256, 128, 64, 3]

        #discriminato config
        disc_layer = [3, 64, 256, 512, 512, 512]

        # -- generator -----
        self.genA = Generator([u'reshape_zA', u'gen_convA', u'gen_deconvA'],
                              gen_conv_layer, gen_deconv_layer)
        self.genB = Generator([u'reshape_zB', u'gen_convB', u'gen_deconvB'],
                              gen_conv_layer, gen_deconv_layer)


        # -- discriminator --
        self.discA = Discriminator([u'disc_convA', u'disc_fcA'], disc_layer)
        self.discB = Discriminator([u'disc_convB', u'disc_fcB'], disc_layer)
        self.lr = 0.00005

        
    def set_model(self):
        # -- place holder --------
        self.figsA= tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3])
        self.figsB= tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3])        
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])

        # -- generators -----------------
        # gen-disc loss
        gen_figsB_from_A = self.genA.set_model(self.figsA, self.z, self.batch_size, True, False)
        gen_loss_from_A = self.discB.set_model(gen_figsB_from_A, True, False)
         
        gen_figsA_from_B = self.genB.set_model(self.figsB, self.z, self.batch_size, True, False)
        gen_loss_from_B = self.discA.set_model(gen_figsA_from_B, True, False)

        # reconstraction error
        re_figA = self.genB.set_model(gen_figsB_from_A, self.z, self.batch_size, True, True)
        figA_recon_error = tf.reduce_sum(tf.abs(self.figsA - re_figA), [1, 2, 3])
        re_figB = self.genA.set_model(gen_figsA_from_B, self.z, self.batch_size, True, True)
        figB_recon_error = tf.reduce_sum(tf.abs(self.figsB - re_figB), [1, 2, 3])
        self.g_obj = tf.reduce_mean(
            - gen_loss_from_A - gen_loss_from_B
            + figA_recon_error + figB_recon_error
        )
        train_var = self.genA.get_variables()
        train_var.extend(self.genB.get_variables())

        self.train_gen  = tf.train.RMSPropOptimizer(self.lr).minimize(self.g_obj, var_list = train_var)
        
        # -- discA --------
        d_lossA = self.discA.set_model(self.figsA, True, True)

        self.d_objA = tf.reduce_mean(-d_lossA + gen_loss_from_B)

        self.train_discA = tf.train.RMSPropOptimizer(self.lr).minimize(self.d_objA, var_list = self.discA.get_variables())

        # -- discB --------
        d_lossB = self.discA.set_model(self.figsB, True, True)

        self.d_objB = tf.reduce_mean(-d_lossB + gen_loss_from_A)

        self.train_discB = tf.train.RMSPropOptimizer(self.lr).minimize(self.d_objB, var_list = self.discB.get_variables())
        
        # -- clipping --------
        c = self.clip_threshold
        var_list = self.discA.get_variables()
        var_list.extend(self.discB.get_variables())
        self.disc_clip = [_.assign(tf.clip_by_value(_, -c, c)) for _ in var_list]
        
        # -- for figure generation -------
        self.gen_figsB_from_A = self.genA.set_model(self.figsA, self.z, self.batch_size, False, True)
        self.gen_figsA_from_B = self.genB.set_model(self.figsB, self.z, self.batch_size, False, True)
        
    def training_gen(self, sess, figsA, figsB, z_list):
        _, g_obj = sess.run([self.train_gen, self.g_obj],
                            feed_dict = {self.figsA: figsA,
                                         self.figsB: figsB,
                                         self.z: z_list})
        return g_obj
        
    def training_disc(self, sess, figsA, figsB, z_list):
        # optimize
        _, d_objA = sess.run([self.train_discA, self.d_objA],
                             feed_dict = {self.z: z_list,
                                          self.figsA:figsA,
                                          self.figsB:figsB})
        _, d_objB = sess.run([self.train_discB, self.d_objB],
                             feed_dict = {self.z: z_list,
                                          self.figsA:figsA,
                                          self.figsB:figsB})
        
        # clipping
        sess.run(self.disc_clip)
        return d_objA, d_objB
    
    def gen_figA(self, sess, figsB, z):
        ret = sess.run(self.gen_figsA_from_B,
                       feed_dict = {self.figsB:figsB, self.z: z})
        return ret
    
    def gen_figB(self, sess, figsA, z):
        ret = sess.run(self.gen_figsB_from_A,
                       feed_dict = {self.figsA:figsA, self.z: z})
        return ret

if __name__ == u'__main__':
    model = Model(z_dim = 30, batch_size = 10, clip_threshold = 0.01)
    model.set_model()
    
