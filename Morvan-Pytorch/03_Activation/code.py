"""
激励函数
    - relu
    - sigmoid
    - tanh
    - softplus
        - softmax也算是激励函数，但不是用来做线图的，是做概率图的。分类问题计算每个类别的概率。
        
激励函数对比
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


# fake data
x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()

# matplotlib can't recognize tensor data, need to transfer to numpy data
# activate function - relu
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()


