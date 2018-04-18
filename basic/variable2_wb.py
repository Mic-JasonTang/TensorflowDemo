#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/19 13:35
# @Author  : Jasontang
# @Site    : 
# @File    : variable2_wb.py
# @ToDo    : 使用convert_variables_to_constants函数将计算图中的变量和取值
# @ToDo    : 通过常量的方式保存，这样整个TensorFlow计算图可以统一存放在一个文件中


import tensorflow as tf
from tensorflow.python.framework import graph_util


v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    # 导出当前计算图的GraphDef部分,只需要这一部分就可以完全从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    # 将图中变量的取值转化为常量，同时将图中不必要的结点去掉。
    # 最后一个参数['add']给出了需要保存的结点的名称,add结点是上面定义的两个变量相加的操作
    # 这里给出的是计算结点的名称，所以没有后面的:0
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    # 将导入的模型存入文件
    with tf.gfile.GFile("variable2_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
