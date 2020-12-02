"""
Dropout 过拟合

解决方案：
1. 增加数据量
2. L1、L2 正规化
        y = Wx
    L1: cost = (Wx - real y)^2 + abs(W)
    L2: cost = (Wx - real y)^2 + (W)^2
    L3,L4... 都类似，换成的三次方和四次方等等。
    W越大，对特定W的惩罚越大。
3. Dropout 正规化 （专门用在神经网络的）
    训练的时候，每次随机忽略掉一些神经网络和神经连接。
    让每次训练的结果都不那么依赖于特定的神经元。
"""
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


torch.manual_seed(1)

N_SAMPLES = 20
N_HIDDEN = 300

# training data
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# test data
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# show data
# plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5)
# plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5)
# plt.legend(loc='upper left')
# plt.ylim((-2.5, 2.5))
# plt.show()

# net
net_overfitting = torch.nn.Sequential(
    nn.Linear(1, N_HIDDEN),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, N_HIDDEN),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, 1)
)

net_dropped = torch.nn.Sequential(
    nn.Linear(1, N_HIDDEN),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, N_HIDDEN),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, 1)
)

print(net_overfitting)
print(net_dropped)

optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net_dropped.parameters(), lr=0.01)
loss_func = nn.MSELoss()

plt.ion()

print(x.size())
print(y.size())

# train
for t in range(500):
    pred_ofit = net_overfitting(x)
    pred_drop = net_dropped(x)
    loss_ofit = loss_func(pred_ofit, y)
    loss_drop = loss_func(pred_drop, y)

    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    loss_ofit.backward()
    loss_drop.backward()
    optimizer_ofit.step()
    optimizer_drop.step()

    if t % 10 == 0:
        # 在预测的时候，不需要dropout，通过 eval 和 train 切换模式
        net_overfitting.eval()
        net_dropped.eval()

        # plotting
        plt.cla()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)
        plt.scatter(x.data.numpy, y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
        plt.scatter(test_x.data.numpy, test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(), fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

        net_overfitting.train()
        net_dropped.train()


plt.ioff()
plt.show()