#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/19 13:53
# @Author  : Jasontang
# @Site    : 
# @File    : variable2_rb.py
# @ToDo    : 使用convert_variables_to_constants函数将计算图中的变量和取值
# @ToDo    : 通过常量的方式保存，这样整个TensorFlow计算图可以统一存放在一个文件中

import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "./variable2_model.pb"
    # 读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将graph_def中保存的图假造到当前的图中.return_elements=["add:0"]给出了返回
    # 的张量名称。在保存的时候给出的计算结点的名称，所以为"add"。在加载的时候给出的是张量
    # 的名称，所以是"add:0"
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    # 输出[array([3.], dtype=float32)]
    print(sess.run(result))


