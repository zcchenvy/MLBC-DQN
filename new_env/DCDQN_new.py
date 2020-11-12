from make_datas import Datas
from memory import Memory
from env import RTB

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import random
GAMMA = 1
R_GAMMA = .75


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self, ad_dim):
        super(Net, self).__init__()
        # B, b, {v}, vw, t
        self.fc1 = nn.Linear(ad_dim*3 + 1, ad_dim * 16)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(ad_dim * 16, ad_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        B = x[:,0:9]
        b = x[:,9:18]
        g_list = x[:,18:27]
        vw = x[:,27:36]
        t = x[:,36:37]

        x = torch.cat((B, b, vw, t), dim=-1)

        x = self.fc1(x)
        x = F.relu(x)
        action_prob = self.out(x)
        action_prob = action_prob.add(g_list)
        return action_prob


class Estimated_Net(nn.Module):
    '''

    Input:
    ------
    x[sum(B), step]

    Output:
    -------
    Estimated Value
    '''

    def __init__(self):
        super(Estimated_Net, self).__init__()
        self.estimated = nn.Linear(10, 1)

    def forward(self, x):
        return self.estimated(x)


class DCDQN():
    """docstring for DQN"""

    def __init__(self, ad_dim, total_step, batch_size=2048, layer=1, interval=10, estimated = True):
        super(DCDQN, self).__init__()

        self.state_dim = ad_dim*4 + 1
        self.ad_dim = ad_dim
        self.total_step = total_step
        # self.memory_capacity = memory_capacity
        # self.q_network_iteration = q_network_iteration
        self.q_network_iteration = total_step * 10
        self.memory_capacity = total_step * 50
        self.estimated = estimated

        self.epsilon = 0.95
        self.epsilon_max = 0.95
        self.epsilon_up = 0.00001
        self.batch_size = batch_size
        self.layer = layer
        self.interval = interval

        if self.estimated:
            self.estimated_net = dict()
            self.estimated_memory = dict()
            self.opt_est = dict()
        else:
            self.target_net = dict()
        self.eval_net = dict()
        self.memory = dict()
        self.optimizer = dict()

        for index in range(self.layer):
            # 部署多层结构
            if self.estimated:
                self.eval_net[index], self.estimated_net[index] = Net(ad_dim), Estimated_Net()
                self.estimated_memory[index] = []
            else:
                self.eval_net[index], self.target_net[index] = Net(ad_dim), Net(ad_dim)

            # 内存池不同，因为B、r和s`不同
            self.memory[index] = np.zeros(
                (self.memory_capacity, (self.state_dim) * 2 + 2))
            # 计数器是通用的

            # 优化器相互独立
            self.optimizer[index] = torch.optim.Adam(
                self.eval_net[index].parameters(), lr=0.001)

            if self.estimated:
                self.opt_est[index] = torch.optim.Adam(
                    self.estimated_net[index].parameters(), lr=0.001)

        self.learn_step_counter = 0
        self.memory_counter = 0
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.loss_func = nn.MSELoss()

    def save(self, args):
        torch.save(self.eval_net[0].state_dict(),
                   'eval_net_{}.pkl'.format(args))

    # 递归策略叠加
    def sort_action(self, state, layer, parent_layer, reset_W):
        copy_state = copy.copy(state)
        discount = layer

        # 处理折扣后的状态
        B = copy_state[0:9]
        b = copy_state[9:18]
        g_list = copy_state[18:27]
        vw = copy_state[27:36]
        t = copy_state[36:37]

        # ！t` = t % interval^layer
        # B` = B / t * t`
        t_ = t % (self.total_step/(self.interval ** layer))

        less_t = t // (self.total_step/(self.interval ** layer))

        B_ = B - reset_W / (self.interval ** layer) * less_t

        new_state = trans_state(B_, b, g_list, vw, t_)
        tensor_state = torch.unsqueeze(
            torch.FloatTensor(new_state), 0)
        action_value = self.eval_net[layer].forward(tensor_state)

        mean_value = np.mean(action_value[0].data.numpy())

        if self.estimated:
            self.estimated_memory[layer].append(np.hstack((B, t, mean_value)))

        sort_value = F.softmax(action_value, dim=1)
        sort_value = sort_value[0].data.numpy()

        if layer == self.layer-1:
            return sort_value

        return sort_value + self.sort_action(state, layer+1, parent_layer, reset_W)*R_GAMMA

    def choose_action(self, state, layer, reset_W, way="train"):
        copy_state = copy.copy(state)
        B = copy_state[0:9]
        b = copy_state[9:18]
        g_list = copy_state[18:27]
        vw = copy_state[27:36]
        t = copy_state[36:37]
        if way == "train":
            if np.random.randn() <= self.epsilon:  # greedy policy
                values = self.sort_action(state, layer, layer, reset_W)

                # 排除所有预算不足的action
                for index in range(len(values)):
                    if b[index] > B[index]:
                        values[index] = -1

                        # next_state = trans_state(B, b, g_list, vw, t-1)
                        # 预算不足也存入内存池，此时action为index，reward为-t
                        # self.store_transition(state = copy_state, action = index, reward = -t, next_state=next_state, layer = layer)

                action = np.argmax(values)

            else:  # random policy
                # 排除预算不足的action进行取值
                actions = []
                for index in range(self.ad_dim):
                    if not b[index] > B[index]:
                        actions.append(index)
                if len(actions) == 0:
                    action = np.random.randint(0, self.ad_dim)
                else:
                    action = random.choice(actions)
        elif way == "test":
            values = self.sort_action(state, layer, layer)
            # 排除所有预算不足的action
            for index in range(len(values)):
                if b[index] > B[index]:
                    values[index] = -1
            action = np.argmax(values)
        return action

    # 经验池中，reward是不同的
    def store_transition(self, state, action, reward, next_state, layer):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_capacity
        self.memory[layer][index, :] = transition
        self.memory_counter += 1

    def learn(self, layer):
        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_up

        # update the parameters
        if self.learn_step_counter % self.q_network_iteration == 0:
            if self.estimated:
                self.press_estimated(layer)
            else:
                self.target_net[layer].load_state_dict(
                    self.eval_net[layer].state_dict())
        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(
            self.memory_capacity, self.batch_size)
        batch_memory = self.memory[layer][sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.state_dim])
        batch_action = torch.LongTensor(
            batch_memory[:, self.state_dim:self.state_dim+1].astype(int))
        batch_reward = torch.FloatTensor(
            batch_memory[:, self.state_dim+1:self.state_dim+2])
        batch_next_state = torch.FloatTensor(
            batch_memory[:, -self.state_dim:])

        batch_next_state_B = torch.FloatTensor(
            batch_memory[:, self.state_dim+2:self.state_dim+2+9])
        batch_next_state_t = torch.FloatTensor(
            batch_memory[:, -1:])

        new_batch_next = torch.cat((batch_next_state_B, batch_next_state_t), dim=1)

        # q_eval
        q_eval = self.eval_net[layer](batch_state).gather(1, batch_action)
        if self.estimated:
            q_next = self.estimated_net[layer](new_batch_next).detach()
        else:
            q_next = self.target_net[layer](batch_next_state).detach()

        # batch_state = torch.FloatTensor(batch_memory[:, :9])
        # q_next = self.estimated_net[layer](batch_next_state).detach()

        q_target = batch_reward + GAMMA * \
            q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer[layer].zero_grad()
        loss.backward()
        self.optimizer[layer].step()

    def press_estimated(self, layer):
        datas = np.asarray(self.estimated_memory[layer])
        s = torch.FloatTensor(datas[:, 0:9])
        t = torch.FloatTensor(datas[:, 9:10])
        v = torch.FloatTensor(datas[:, 10:11])

        ss = torch.cat((s, t), dim=1)
        for _ in range(50):
            v_ = self.estimated_net[layer](ss)
            loss = self.loss_func(v_, v)

            self.opt_est[layer].zero_grad()
            loss.backward()
            self.opt_est[layer].step()

        self.estimated_memory[layer] = []


def trans_state(B, b, g_list, vw, t):
    # 预算不足的，g_list改为-t
    # for index in range(len(g_list)):
    #     if B[index] < b[index]:
    #         g_list[index] = -t
    return np.hstack((B, b, g_list, vw, t))


def main():
    import random

    random.seed(10)

    with open('./pickles/datas.pickle', "rb") as f:
        datas = pickle.load(f)

    layer_count = 1
    interval = 10
    total_step = 1000

    # 100000 10000 1000 100

    env = dict()

    for layer in range(layer_count):
        env[layer] = RTB(datas=datas, total_step=total_step //
                         (interval**layer)-1)

    dqn = DCDQN(ad_dim=9, total_step=total_step,
                layer=layer_count, interval=interval, estimated = True)

    episodes = 10000
    print("Collecting Experience....")
    reward_list = []
    total_price = []
    # plt.ion()
    # fig, ax = plt.subplots()

    # 因为python对多线程支持不是很友好，所以训练交替进行/论文中可写多线程并行

    for i in range(episodes):
        for layer in range(layer_count-1, -1, -1):
            if layer == 0:
                if i % 10 == 0:
                    [W, w, v, vw, t], done = env[layer].reset()
                    state = trans_state(W, w, v, vw, t)
                    while not done:
                        for index in range(env[layer].m):
                            if W[index] < w[index]:
                                vw[index] = 0
                        action = np.argmax(vw)
                        [W_, w_, v_, vw_, t_], reward, done = env[layer].step(
                            action)
                        next_state = trans_state(W_, w_, v_, vw_, t_)
                        dqn.store_transition(
                            state, action, reward, next_state, layer=layer)
                        if done:
                            break
                        state = next_state
                        W, w, v, vw, t = W_, w_, v_, vw_, t_
                # 主线程保存实验结果
                [W, w, v, vw, t], done = env[layer].reset()
                state = trans_state(W, w, v, vw, t)
                ep_reward = 0
                
                while not done:
                    # 主线程
                    # print("step: {}/{}, layer: {}, episodes:{}/{}".format(t,
                    #                                                       env[layer].total_step, layer, i+1, episodes), end="\r")
                    action = dqn.choose_action(
                        state, layer, env[layer].reset_W)

                    [W_, w_, v_, vw_, t_], reward, done = env[layer].step(
                        action)
                    next_state = trans_state(W_, w_, v_, vw_, t_)

                    dqn.store_transition(
                        state, action, reward, next_state, layer=layer)
                    ep_reward += reward

                    if dqn.memory_counter >= dqn.memory_capacity/2:
                        dqn.learn(layer)
                        if done:
                            print("episode: {} , the episode reward is {}, epsilon: {}".format(
                                i, round(ep_reward, 3), dqn.epsilon))
                            print("episode: {}/{}, 回报: {}, CTR期望: {}, 平台总收益: {}".format(
                                i+1, episodes, round(ep_reward, 3), env[layer].total_v, env[layer].total_w))
                    if done:
                        break
                    state = next_state
                    W, w, v, vw, t = W_, w_, v_, vw_, t_
                r = copy.copy(ep_reward)
                reward_list.append(r)
                total_price.append(env[layer].total_w)

                # 保存模型
                if i % 100 == 0:
                    dqn.save(i)

                np.savetxt('total_price.csv', total_price, delimiter=',')
                np.savetxt('reward_list.csv', reward_list, delimiter=',')
            else:
                loops = 10 ** layer
                # 非主线程 只做训练不进行打印 由于采样低 所以采样次数增加
                for loop in range(loops):
                    if i % 10 == 0:
                        if i % 10 == 0:
                            [W, w, v, vw, t], done = env[layer].reset()
                            state = trans_state(W, w, v, vw, t)
                            while not done:
                                for index in range(env[layer].m):
                                    if W[index] < w[index]:
                                        vw[index] = 0
                                action = np.argmax(vw)
                                [W_, w_, v_, vw_, t_], reward, done = env[layer].step(
                                    action)
                                next_state = trans_state(W_, w_, v_, vw_, t_)
                                dqn.store_transition(
                                    state, action, reward, next_state, layer=layer)
                                if done:
                                    break
                                state = next_state
                                W, w, v, vw, t = W_, w_, v_, vw_, t_
                    [W, w, v, vw, t], done = env[layer].reset()
                    state = trans_state(W, w, v, vw, t)

                    while not done:
                        print("step: {}/{}, layer: {}, episodes:{}/{}".format(t,
                                                                              env[layer].total_step, layer, i+1, episodes), end="\r")
                        action = dqn.choose_action(
                            state, layer, env[layer].reset_W)

                        [W_, w_, v_, vw_, t_], reward, done = env[layer].step(
                            action)
                        next_state = trans_state(W_, w_, v_, vw_, t_)

                        dqn.store_transition(
                            state, action, reward, next_state, layer=layer)

                        if dqn.memory_counter >= dqn.memory_capacity/2:
                            dqn.learn(layer)
                        if done:
                            break
                        state = next_state
                        W, w, v, vw, t = W_, w_, v_, vw_, t_


def test():
    with open('./pickles/datas.pickle', "rb") as f:
        datas = pickle.load(f)

    layer_count = 2
    interval = 10
    total_step = 10000
    layer = 0
    env = {}
    env[layer] = RTB(datas=datas, total_step=total_step-1)
    dqn = DCDQN(ad_dim=9, total_step=total_step,
              layer=layer_count, interval=interval)
    dqn.eval_net[0].load_state_dict(torch.load('eval_net_1100.pkl'))
    [W, w, v, vw, t], done = env[layer].reset()
    state = trans_state(W, w, v, vw, t)
    ep_reward = 0

    while not done:
        # 主线程
        print("step: {}/{}, layer: {}, episodes:{}/{}".format(t,
                                                              env[layer].total_step, layer, 0, 1), end="\r")
        action = dqn.choose_action(state, layer, way='test')
        # action = np.argmax(vw)

        [W_, w_, v_, vw_, t_], reward, done = env[layer].step(
            action)
        next_state = trans_state(W_, w_, v_, vw_, t_)

        dqn.store_transition(
            state, action, reward, next_state, layer=layer)
        ep_reward += reward

        if done:
            print("回报: {}, CTR期望: {}, 平台总收益: {}".format(
                round(ep_reward, 3), env[layer].total_v, env[layer].total_w))
            break
        state = next_state
        W, w, v, vw, t = W_, w_, v_, vw_, t_


if __name__ == "__main__":
    main()
    # test()
