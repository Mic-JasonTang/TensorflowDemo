#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/17 20:24
# @Author  : Jasontang
# @Site    : 
# @File    : saver_demo2.py
# @ToDo    : 反序列化模型


import tensorflow as tf

'''
# # # 变量恢复

# 使用和保存模型代码中一样的方式来声明变量
# 需要保证变量的属性和存储的时候是一样的
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0), name="v2")

result = v1 + v2

saver = tf.train.Saver()

# 如果只是想加载部分变量，比如这里就只加载v1,没有加载v2就会报错
# FailedPreconditionError (see above for traceback): Attempting to use uninitialized value v2
saver = tf.train.Saver([v1])

with tf.Session() as sess:
    # 加载已经保存的模型,并通过已经保存的模型中的变量的值来计算加法
    saver.restore(sess, "./model_saver_demo/saver_demo.ckpt")
    print(sess.run(result))
'''


'''
# # # 直接恢复计算图恢复

# 如果不希望重复定义图上的运算,也可直接加载已经持久化的图
# 加载了计算图上定义的全部变量
saver = tf.train.import_meta_graph("./model_saver_demo/saver_demo.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "./model_saver_demo/saver_demo.ckpt")
    # 通过张量的名称来获取变量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
'''

# # # 恢复时,变量重命名

# 如果在恢复时,声明的变量和已经保存模型中的变量的名称不同
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
# 这里虽然声明了变量的值为5，但是在恢复时还是会变为已经存储的2.0
v2 = tf.Variable(tf.constant(5.0), name="other-v2")

result = v1 + v2
# 如果直接使用tf.train.Saver()来加载模型会报变量找不到的错误
# NotFoundError (see above for traceback): Key other-v1 not found in checkpoint
# saver = tf.train.Saver()

# 使用一个字典来重命名变量就可以加载原来的模型,
saver = tf.train.Saver({"v1": v1, "v2": v2})

with tf.Session() as sess:
    saver.restore(sess, "./model_saver_demo/saver_demo.ckpt")
    saver.export_meta_graph("./model_saver_demo/saver_demo.ckpt.meta.json", as_text=True)
    print(sess.run(result))