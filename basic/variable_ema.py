#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/18 22:43
# @Author  : Jasontang
# @Site    : 
# @File    : variable_ema.py
# @ToDo    : 变量重命名的目的之一: 使用变量的滑动平均值
# 如果在加载模型时直接将影子变量映射到变量自身，那么在使用训练好的模型时就
# 不需要再调用函数来获取变量的滑动平均值了


import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
# 在没有申明滑动平均模型时只有一个变量v,所以下面的语句只会输出"v:0"

for variable in tf.global_variables():
    print(variable.name)

print("-" * 20)

# 使用滑动平均,rate*shadow_variable+(1-rate)*variabel
ema = tf.train.ExponentialMovingAverage(0.99)
# 定义一个更新变量滑动平均的操作。每次执行这个操作时这个列表中的变量都会被更新。
maintain_average_op = ema.apply(tf.global_variables())
# 在声明滑动平均模型之后，tf会自动生成一个影子变量v/ExponentialMovingAverage
for variable in tf.global_variables():
    # 输出:
    # v:0
    # v/ExponentialMovingAverage:0
    print(variable.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 更新v=10
    sess.run(tf.assign(v, 10))
    sess.run(maintain_average_op)
    # 在这里会将v:0和v/ExponentialMovingAverage:0都存下来
    saver.save(sess, './model_variable_ema/variable_ema.ckpt')
    # 输出[10.0, 0.099999905]
    # ema.average()是回去滑动平均之后变量（影子变量）的取值
    print(sess.run([v, ema.average(v)]))

