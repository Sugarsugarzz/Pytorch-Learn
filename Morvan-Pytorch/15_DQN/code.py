"""
Reinforcement Learning 强化学习
DQN
Q-Learning
    - Experience replay
    - Fixed Q-targets

强化学习：不经过监督，让其自己学习做某些事情。
    游戏：让杆子立起来！
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym  # 强化学习库

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9  # greedy policy
GAMMA = 0.9    # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 动作
N_STATES = env.observation_space.shape[0]  # 状态


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # 采取动作
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        # 记忆库，存储之前的记忆，学习过程从记忆库提取记忆然后再进一步学习
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 学习过程
        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        # eval net update always
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES: N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1: N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
# 整个强化学习的过程
for i_episode in range(400):
    s = env.reset()  # 所处环境反馈
    while True:
        env.render()  # 环境渲染

        a = dqn.choose_action(s)  # 根据现在的状态来采取行动

        # take action
        s_, r, done, info = env.step(a)  # 根据采取的行为做一些反馈

        # modify the reward


        dqn.store_transition(s, a, r, s_)  # 存储现在的反馈

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:
            break
        s = s_
