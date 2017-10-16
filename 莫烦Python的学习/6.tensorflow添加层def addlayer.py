# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/10/16 16:25'
__site__ = ''
__software__ = 'PyCharm'
__file__ = '6.tensorflow添加层def addlayer.py'

import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_fun=None):
    Weights = tf.Variable(tf.random_uniform([in_size, out_size]))  # 我们在这里定义的时候，如果变量是矩阵，我们就大写首字母
    biases = tf.Variable(tf.zeros(1, out_size) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_fun is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_fun(Wx_plus_b)
    return outputs


if __name__ == '__main__':
    pass
