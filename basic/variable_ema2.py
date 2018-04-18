#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/18 23:35
# @Author  : Jasontang
# @Site    : 
# @File    : variable_ema2.py
# @ToDo    : 变量重命名的目的之一: 使用变量的滑动平均值
# 直接恢复变量的影子变量的值

import tensorflow as tf


v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)
# 输出
# {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
print(ema.variables_to_restore())

# 通过变量重命名将原来变量v的滑动平均值直接赋值给v
# saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
saver = tf.train.Saver(ema.variables_to_restore())

with tf.Session() as sess:
    # saver.restore(sess, "./model_variable_ema/variable_ema.ckpt")
    saver.export_meta_graph("./model_variable_ema/variable_ema.ckpt.meta.json", as_text=True)
    # 输出0.099999905,即原模型中v的滑动平均值
    # print(sess.run(v))
