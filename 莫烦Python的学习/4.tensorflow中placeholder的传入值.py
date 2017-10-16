# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/10/16 14:21'
__site__ = ''
__software__ = 'PyCharm'
__file__ = '4.tensorflow中placeholder的传入值.py'

import tensorflow as tf

input_1 = tf.placeholder(tf.float32)
input_2 = tf.placeholder(tf.float32)

output = tf.multiply(input_1, input_2)

with tf.Session() as sess:
    print(sess.run(output,
                   feed_dict={input_1:[7.], input_2:[2.]}))
