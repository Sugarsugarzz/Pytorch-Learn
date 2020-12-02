"""
AutoEncoder 自编码 神经网络的非监督学习

    原图片 - 打码图片 - 原图片
    压缩  ->  解压

    高清图片，数据量可能达到上千万，神经网络学习压力大。
    压缩一下，从图片中提取最具代表性的信息，将缩减过后的信息交给神经网络学习，就变得轻松了。

    自编码：将原数据压缩，并解压。通过对比压缩解压前后的求出误差，进行反向传递，逐步提升自编码的准确性。
        中间部分就是原数据的精髓，
        由于只用到了原数据，而没有用到标签，所以可以说是一种非监督学习。

    通过自编码学习原数据的精髓，然后让神经网络学习精髓部分，就减轻了神经网络学习的压力，并达到同样好的效果。

    自编码在提取主要特征上，甚至超越了 PCA，给特征属性降维。

    简单说，有一个Encoder和一个Decoder，将数据压缩和解压的过程。
        Encoder将输入压缩成小的数据集，然后用压缩的精髓部分代表特征。
        Decoder

    ---
    这里不需要测试数据，用训练数据自身来做对比
"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data as Data
import torchvision
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

train_data = torchvision.datasets.MNIST(
    root='../10_CNN/mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# build network
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()   # 将输出压缩到 0~1 的范围
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoEncoder = AutoEncoder()
optimizer = torch.optim.Adam(autoEncoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()
plt.show()

# original data (first row) for viewing
view_data = Variable(train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor))
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28 * 28))  # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1, 28 * 28))  # batch y, shape (batch, 28*28)  注意这里是x数据
        b_label = Variable(y)                # batch label

        encoded, decoded = autoEncoder(b_x)

        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch: ', epoch, ' | step: ', step, ' | train loss: %.4f' % loss.data[0])

            # plot decoded image (second row)
            _, decoded_data = autoEncoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()