"""
分类
    代码和回归的类似，数据要换
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)  # 二维坐标
y0 = torch.zeros(100)  # label
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x, y = Variable(x), Variable(y)
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1])
plt.show()

# define neural network
class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.predict(x)
        return x

# # build network complete
net = Net(2, 10, 5)
print(net)
plt.ion()
plt.show()

# optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵，用于分类问题

# train
for t in range(100):
    out = net(x)

    loss = loss_func(out, y)

    optimizer.zero_grad()  # 将神经网络中的参数的梯度先降为0，避免爆炸。因为每次计算loss，梯度都会保留在optimizer和net里面，所以要先梯度设为0
    loss.backward()
    optimizer.step()

    # visualize
    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0)
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

    plt.ioff()
    plt.show()



