import torch
print(torch.__version__)
from tqdm import tqdm

for char in tqdm(['a', 'b', 'c', 'c']):
    pass

a = ['a', 'b', 'c']
b = ['a', 'd', 'e']
c = [a, b]



""" 中文文本分割 """
sentence = "哈哈，哈哈哈。呵呵，呵呵呵！！怎么回事？呃；无语了……"


def split_sentence(sentence):
    punctuations = ['。', '！', '；', '？', '……']
    sent_list = []
    index = 0
    flag, flag2 = False, False
    for i, ch in enumerate(sentence):
        if ch in punctuations:
            flag = True
        if ch not in punctuations and flag:
            flag = not flag
            flag2 = True
        if flag2:
            sent_list.append(sentence[index:i])
            index = i
            flag2 = not flag2
    if index < len(sentence):
        sent_list.append(sentence[index:])
    return sent_list

print(split_sentence(sentence))