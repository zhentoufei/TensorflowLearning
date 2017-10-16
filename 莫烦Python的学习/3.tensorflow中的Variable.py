# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/10/16 14:11'
__site__ = ''
__software__ = 'PyCharm'
__file__ = '3.tensorflow中的Variable.py'

import tensorflow as tf

state = tf.Variable(0, name='counter') # 在tensorflow中只有定义了他是变量，他才是变量
print(state.name) # counter:0

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables() # 如果定义了变量，那么必须要有这句话

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
