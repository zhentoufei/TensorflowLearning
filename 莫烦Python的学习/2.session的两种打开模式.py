# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/10/16 13:56'
__site__ = ''
__software__ = 'PyCharm'
__file__ = '2.py'

import tensorflow as tf
matrix_1 = tf.constant([[3,3]])
matrix_2 = tf.constant([[2],
                        [2]])

product = tf.matmul(matrix_1, matrix_2) #matric multiply np.dot(m1, m2)

# method 1
# sess = tf.Session()
# res = sess.run(product)
# print(res)
# sess.close()

# method 2 这种方法下会自动close相应的session
with tf.Session() as sess:
    res_2 = sess.run(product)
    print(res_2)