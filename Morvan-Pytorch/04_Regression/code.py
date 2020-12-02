"""
回归：连续的值
用线段拟合到离散数据点上
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# fake data
# torch只会处理二维的数据，需要升维
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

x, y = Variable(x), Variable(y)
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# define neural network
class Net(torch.nn.Module):
    """
    继承torch.nn.Module模块，很多功能包含在这个模块里
    init和forward是最需要的。
        init：搭建层所需要的信息
        forward：前向传递的过程，层信息放在forward中一个一个组合起来
    """
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

# build network complete
net = Net(1, 10, 1)
print(net)
plt.ion()
plt.show()

# optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()  # 均方差

# train
for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()  # 将神经网络中的参数的梯度先降为0，避免爆炸。因为每次计算loss，梯度都会保留在optimizer和net里面，所以要先梯度设为0
    loss.backward()
    optimizer.step()

    # visualize
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

    plt.ioff()
    plt.show()



