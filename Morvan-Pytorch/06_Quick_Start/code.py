"""
快速搭建
    激励函数作为一个层结构。
    效果都是一样的。
"""
import torch

net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
print(net)