#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from tf_util import conv, deconv, linear, Layers, lrelu, batch_norm, get_dim

class Generator(Layers):
    def __init__(self, name_scopes, en_channels, dec_channels):
        assert(len(name_scopes) == 3)
        assert(len(en_channels) - 1 == len(dec_channels))

        super().__init__(name_scopes)
        self.en_channels = en_channels
        self.dec_channels = dec_channels        

    def set_model(self, input_figs, z, batch_size, is_training = True, reuse = False):
        assert(self.en_channels[0] == input_figs.get_shape().as_list()[3])
        
        # reshape z
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            h = linear('_r', z, get_dim(input_figs))
            h = batch_norm('reshape', h, decay_rate= 0.99,
                           is_training = is_training)
            h = tf.nn.relu(h)
        height = input_figs.get_shape().as_list()[1]
        width = input_figs.get_shape().as_list()[2]        
        h = tf.reshape(h, [-1, height, width, self.en_channels[0]])
        h = tf.concat([h, input_figs], 3)
        
        # convolution
        encoded_list = []
        
        # encode
        with tf.variable_scope(self.name_scopes[1], reuse = reuse):
            for i, out_dim in enumerate(self.en_channels[1:]):
                h = conv(i, h, out_dim, 4, 4, 2)
                if i == 0:
                    encoded_list.append(h)
                    h = lrelu(h)
                else:
                    h = batch_norm(i, h, 0.99, is_training)
                    encoded_list.append(h)
                    h = lrelu(h)
                    
        # deconvolution
        encoded_list.pop()
        h = tf.nn.relu(h)
        
        with tf.variable_scope(self.name_scopes[2], reuse = reuse):
            for i, out_chan in enumerate(self.dec_channels[:-1]):
                # get out shape
                h_shape = h.get_shape().as_list()
                out_width = 2 * h_shape[2] 
                out_height = 2 * h_shape[1]
                out_shape = [batch_size, out_height, out_width, out_chan]
                
                # deconvolution
                deconved = deconv(i, h, out_shape, 4, 4, 2)

                # batch normalization
                h = batch_norm(i, deconved, 0.99, is_training)
                if i <= 2:
                    h = tf.nn.dropout(h, 0.5)
                h = tf.concat([h, encoded_list.pop()], 3)
                # activation
                h = tf.nn.relu(h)
            height = 2 * h.get_shape().as_list()[1]
            width = 2 * h.get_shape().as_list()[1]
            out_shape = [batch_size, height, width, self.dec_channels[-1]]
            h = deconv(i + 1, h, out_shape, 4, 4, 2)
        return tf.nn.tanh(h)
        
    
if __name__ == u'__main__':
    g = Generator([u'reshape_z', u'convolution', u'deconvolution'],
                  [3, 64, 128, 256, 512, 512, 512, 512, 512],
                  [512, 512, 512, 512, 256, 128, 64, 3])
    z = tf.placeholder(tf.float32, [None, 100])
    figs = tf.placeholder(tf.float32, [None, 256, 256, 3])    
    h = g.set_model(figs, z, 10)
    h = g.set_model(figs, z, 10, True, True)    
    print(h)
