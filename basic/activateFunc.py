
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/2 19:45
# @Author  : Jasontang
# @Site    : 
# @File    : activateFunc.py
# @ToDo    : 常用激活函数图像

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x1 = np.arange(0, 10, 0.0001)
print(x1)
fx1 = 1.0 / (1 + np.exp(-x1))
fx2 = (1.0 - np.exp(-2*x1)) / (1 + np.exp(-2*x1))
fx3 = (-np.log(x1))

# plt.plot(x1, fx1, label="sigmoid")
# plt.plot(x1, fx2, label="tanh")
plt.plot(x1, fx3, label="-log")
plt.legend()
plt.show()
