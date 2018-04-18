#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/21 20:41
# @Author  : Jasontang
# @Site    : 
# @File    : main.py
# @ToDo    : 驱动程序

import _thread

from neural_network_learning.hand_writting_refactor import mnist_train, mnist_eval


if __name__ == '__main__':
    _thread.start_new_thread(mnist_train.main, (None,))
    _thread.start_new_thread(mnist_eval.main, (None,))

    # 这个不能删除，当做主线程
    while 1:
        pass








