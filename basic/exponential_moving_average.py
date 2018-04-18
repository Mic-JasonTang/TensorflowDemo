#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/4 17:21
# @Author  : Jasontang
# @Site    : 
# @File    : exponential_moving_average.py
# @ToDo    : 滑动平均模型 - 使得模型在测试数据上更健壮

import tensorflow as tf

# 定义了一个变量用于计算滑动平均，这个变量的初始值为0，注意这里手动制定了额变量的类型为tf.float32，
# 因为所有需要计算活动平均的变量必须是实数型
v1 = tf.Variable(0, dtype=tf.float32)
# 这里num_updates变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率
num_updates = tf.Variable(0, trainable=False)
# 定义一个滑动平均的类，初始化时给定了衰减率0.99,和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, num_updates)
# 定义一个更新变量滑动平均的操作，这里需要给定一个变量列表，每次执行这个操作时，
# 这个列表中的变量都会被更新
maintain_average_op = ema.apply([v1])

with tf.Session() as sess:

    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 通过ema.average(v1)获取滑动平均之后变量的取值。在初始化之后变量v1的值和v1的滑动平均都为0
    print(sess.run((v1, ema.average(v1))))
    # 更新变量v1的值为5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值。衰减率更新公式为min{0.99, (1+step)/(10+step)=0.1} = 0.1
    # 所以v1的滑动平均会被更新为0.1 * 0 + 0.9 * 5 = 4.5
    sess.run(maintain_average_op)
    print(sess.run((v1, ema.average(v1))))

    # 更新step的值为1000
    sess.run(tf.assign(num_updates, 10000))
    # 更新v1的值为10
    sess.run(tf.assign(v1, 10))
    # 更新滑动平均的值，衰减率更新公式为min{0.99, (1+step)/(10+step) = 0.999} = 0.99
    # 所以v1更新为 0.99 * 4.5 + 0.01 * 10 = 4.555
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
    # 更新滑动平均的值，衰减率更新公式为min{0.99, (1+step)/(10+step) = 0.999} = 0.99
    # 所以v1更新为 0.99 * 4.555 + 0.01 * 10 = 4.60945
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
    # 更新滑动平均的值，衰减率更新公式为min{0.99, (1+step)/(10+step) = 0.999} = 0.99
    # 所以v1更新为 0.99 * 4.60945 + 0.01 * 10 = 4.6633554
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))

