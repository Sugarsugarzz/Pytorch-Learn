"""
批训练
    Torch提供一个整理数据结构的东西 - DataLoader！
    可以用其来包装数据，进行批训练，而且批训练可以有很多种途径。

    数据量非常大，可以一小批一小批的训练，快速的提升神经网络的效果、效率。
    minibatch training 集合了批梯度下降和随机梯度下降的有点。
"""
import torch
import torch.utils.data as Data

BATCH_SIZE = 5  # 如果是8，而数据集size是10，则第一次会提取8个，第二次提取2个
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)  # data_tensor and target_tensor
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# 3个epoch，BATCH_SIZE为5，所以每个epoch训练2次
for epoch in range(3):
    # 拆分成2组时，可以定义loader是否要打乱这批数据 - shuffle
    for step, (batch_x, batch_y) in enumerate(loader):
        # training...
        print('Epoch: ' , epoch, ' | Step: ', step, ' | batch x: ',
              batch_x.numpy(), ' | batch y: ', batch_y.numpy())