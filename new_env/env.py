import numpy as np 
from make_datas import Datas
import pickle
import copy
import matplotlib.pyplot as plt 

'''
    模型的更新主要是为了使得模型的应用场景更加泛化

    每种商品具有 v(ctr) w(bit)

    一共有n个商品，m个背包

    每种商品的出现概率为p

    数据的加载通过文件，文件中包含上述的内容

    m个背包中，每个背包的质量为W(B)，当然，这个W是随机的
'''
class RTB:
    def __init__(self, datas, total_step):
        self.datas = datas

        self.m = datas.m
        self.n = datas.n
        self.v = datas.v*1000
        self.w = datas.w*1000
        self.p = datas.p
        self.total_step = total_step

        if not self.check_datas():
            exit()
        
        # 预计总收益v
        self.total_v = None
        # 预计总消耗w
        self.total_w = None
        self.W = None
        self.reset_W = None
        # 当前商品编号
        self.number = None

        # 出价序列
        self.bit_list = None
        self.value_list = None

    def check_datas(self):
        '''
            检查数据的完整性
        '''
        if not self.v.shape[0] == self.n:
            print("Data Error: v shape 0 is not equal n")
            return False
        if not self.v.shape[1] == self.m:
            print("Data Error: v shape 1 is not equal m")
            return False
        if not self.w.shape[0] == self.n:
            print("Data Error: w shape 0 is not equal n")
            return False
        if not self.w.shape[1] == self.m:
            print("Data Error: w shape 1 is not equal m")
            return False
        if not self.p.shape[0] == self.n:
            print("Data Error: p length is not equal n")
            return False
        if not sum(self.p) <= 1:
            print("Data Error: total p is not equal 1")
            return False
        return True

    def reset(self):
        self.total_v = 0
        self.total_w = 0

        self.t = self.total_step

        self.W = np.asarray([self.total_step*1000*0.0005] * self.m)
        random_W = np.random.rand(self.m)
        random_W = random_W/sum(random_W)
        self.W *= random_W

        self.reset_W = copy.copy(self.W)

        self.refresh_number()

        self.bit_list = []
        self.value_list = []

        return self.get_state(), False

    def refresh_number(self):
        self.number = np.argmax(np.random.multinomial(1, self.p))

    def get_state(self):
        W = copy.copy(self.W)
        w = copy.copy(self.w[self.number])
        v = copy.copy(self.v[self.number])
        vw = v/w
        t = copy.copy(self.t)
        return [W, w, v, vw, t]

    def step(self, action):
        done = False
        income_v = 0
        if not action in range(0, self.m):
            print("Action Waring: maybe agent give a wrong action!")
        if self.W[action] >= self.w[self.number][action]:
            self.W[action] -= self.w[self.number][action]
            self.total_v += self.v[self.number][action]
            self.total_w += self.w[self.number][action]
            # income_v = (self.v[self.number][action] - np.mean(self.v[self.number])) ** 3
            income_v = self.v[self.number][action]
            self.bit_list.append(self.v[self.number][action])
            self.value_list.append(self.w[self.number][action])
        else:
            self.bit_list.append(0)
            self.value_list.append(0)
            income_v = - self.t
            # income_v = 0
        self.t -= 1
        if self.t == 0:
            done = True
        
        self.refresh_number()
        return self.get_state(), income_v, done

def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

if __name__ == "__main__":
    with open('./pickles/datas.pickle', "rb") as f:
        datas = pickle.load(f)
    
    # np.savetxt("v.csv", datas.v, delimiter=',')
    # np.savetxt("w.csv", datas.w, delimiter=',')
    # np.savetxt("p.csv", datas.p, delimiter=',')

    # vw = datas.v/datas.w

    # np.savetxt("vw.csv", vw, delimiter=',')

    # 环境测试
    env = RTB(datas, total_step=1000)
    [W, w, v, vw, t], done = env.reset()
    reward = 0

    while not done:
        print('step: {}/{}'.format(env.t, env.total_step), end="\r")
        
        # 清理不足的  决策选择最高性价比
        for index in range(env.m):
            if W[index] < w[index]:
                vw[index] = 0
        action = np.argmax(vw)

        # 清理不足的  决策选择最优的ctr
        # for index in range(env.m):
        #     if W[index] < w[index]:
        #         v[index] = 0
        # action = np.argmax(v)

        

        # # 清理不足的    决策选择最优的bit
        # for index in range(env.m):
        #     if W[index] < w[index]:
        #         w[index] = 0
        # action = np.argmax(w)

        [W, w, v, vw, t], income, done = env.step(action)
        reward += income
        # print(w)
        # print(W)
        if done:
            print('total v is {}, total w is {}'.format(env.total_v, env.total_w))
            bits = list_split(env.bit_list, 10)
            values = list_split(env.value_list, 10)
            plt.plot(np.sum(bits,axis=1), label='bit')
            plt.plot(np.sum(values,axis=1), label='CTR')
            plt.legend()
            # plt.show()
    print(reward)