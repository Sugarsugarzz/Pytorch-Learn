"""
RNN：
    在有序数据上学习。
    梯度消失：w小于1，传递到最后，w接近0
    梯度爆炸：w大于1，传递到最后，w无限大

LSTM（长短期记忆）：
    相比RNN多了三个控制器：输入、输出、忘记。

Classification
手写数字识别
"""
import torch
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 64
# RNN 每次读取一行，一行有28个像素点，总共有28行
TIME_STEP = 28  # rnn time step / image height
INPUT_SIZE = 28  # rnn input size / image width
LR = 0.01
DOWNLOAD_MNIST = True

train_data = dsets.MNIST(root='../10_CNN/mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = dsets.MNIST(root='../10_CNN/mnist', train=False, transform=transforms.ToTensor)
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(  # LSTM比nn.RNN效果好
            input_size=INPUT_SIZE,  # 表示每行数据像素点个数
            hidden_size=64,  # 表示隐藏层节点个数
            num_layers=1,  # 表示隐藏层的层数
            batch_first=True,  # (batch, time_step, input)，将batch放在第一个维度
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # hidden status，None表示没有初始 hidden state
        # chosse r_out at the last time step
        out = self.out(r_out[:, -1, :])  # (batch, time_step, input)
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# training
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28, 28)

        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output,  1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size
            print('Epoch: ', epoch, '| Step ', step, ' | train loss: %.4f' % loss.data[0], ' | test accuracy: %.4f' % accuracy)


# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, ' prediction number')
print(test_y[:10], ' real number')




