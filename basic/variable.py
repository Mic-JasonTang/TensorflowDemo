#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/15 13:33
# @Author  : Jasontang
# @Site    : 
# @File    : variable.py
# @ToDo    : 变量管理

import tensorflow as tf
'''
# 通过下面两种方式创建的变量是一样的
v2 = tf.Variable(tf.constant(0.0, shape=[1], name='v'))
# tf.get_varibale()首先会试图创建一个名字为v的参数，如果创建失败(比如已经有同名的参数)，那么就会报错
# ValueError: Variable v already exists, disallowed. Did you mean to set reuse=True
# or reuse=tf.AUTO_REUSE in VarScope?
# v2 = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer)
v1 = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer)
with tf.Session() as sess:
    sess.run(v2.initializer)
    print(v2)
    sess.run(v1.initializer)
    print(v1)

# 使用命名空间来创建变量
# 在foo命名空间创建名字为v的变量
with tf.variable_scope("foo"):
    v = tf.get_variable("v", shape=[1,2], initializer=tf.constant_initializer)

# 因为在命名空间foo中已经存在名字为v的变量，下面代码会报错
# with tf.variable_scope("foo"):
#     v = tf.get_variable("v", shape=[2])

# 在生成上下文管理器时，将参数reuse设置为True，这样tf.get_variable函数将直接获取已经声明的变量
with tf.variable_scope("foo", reuse=True):
    # 这里要相等时，必须保持shape也相同, 如果不指定shape,则会自动调整为上述变量的shape
    v1 = tf.get_variable("v")
    print(v1 == v)
'''
# 将参数reuse设置为True时，tf.variable_scope将只能获取已经创建过的变量，否则报错
# ValueError: Variable bar/v does not exist, or was not created with tf.get_variable().
# Did you mean to set reuse=tf.AUTO_REUSE in VarScope?
# with tf.variable_scope("bar", reuse=True):
#     v = tf.get_variable("v", [1])


# 嵌套使用上下文管理器
with tf.variable_scope("root"):
    # 可以通过tf.get_variable_scope().reuse函数唉获取当前上下文管理器中reuse参数
    # 的取值. 输出False,最外层的reuse默认是None/False
    print(tf.get_variable_scope().reuse)

    with tf.variable_scope("foo", reuse=True):  # 新建一个嵌套的上下文管理器,resue指定为True

        print(tf.get_variable_scope().reuse)    # 输出为True

        with tf.variable_scope("bar"):          # 新建一个嵌套的上下文管理器,reuse的取值会和外面一层保持一致
            print(tf.get_variable_scope().reuse)   # 输出为True

    print(tf.get_variable_scope().reuse)        # 输出为False,退出reuse设置为True的上下文之后,reuse的值又回到了False


# 提供一个管理变量命名空间的方式
v1 = tf.get_variable("v", [1])
# 输出v:0. "v"为变量名称, "0"表示这个变量是生成变量这个运算的第一个结果
print(v1.name)

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v", [1])
    # 输出foo/v:0. 在tf.variable_scope中创建的变量，名称前面会加入命名空间的名称，并通过/来分隔命名空间的名称和变量的名称
    print(v2.name)

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        # 输出foo/bar/v:0. 明明靠空间可以嵌套,同时变量的名称也会加入所有命名空间的名称作为前缀
        print(v3.name)

    v4 = tf.get_variable("v1", [1])
    # foo/v1:0. 当命名空间退出后，变量名称也就不会再被加入其前缀了
    print(v4.name)

# 创建一个名称为空的命名空间,并设置reuse=True
with tf.variable_scope("", reuse=True):
    # 可以直接通过带命名空间名称的变量名来获取其他命名空间下的变量。
    # 比如这里通过指定名称foo/bar/v来获取在命名空间foo/bar/中创建的变量
    v5 = tf.get_variable("foo/bar/v", [1])

    print(v5 == v3)  # 输出True
    v6 = tf.get_variable("foo/v1", [1])
    print(v6 == v4)  # 输出True
