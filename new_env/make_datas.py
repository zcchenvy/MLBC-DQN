from datas.load_data import load_LoadData
from datas.load_data import LoadData

import numpy as np
import pickle
import os

class Datas:
    def __init__(self):
        datas = load_LoadData(columns=['weekday', 'hour', 'region'])

        n = datas.count
        m = len(datas.files)
        v = np.zeros([n, m])
        w = np.zeros([n, m])
        p = np.zeros(n)

        for M in range(m):
            file_name = datas.files[M]
            for N in range(n):
                if datas.columns_counts[file_name][N] == 0 or datas.columns_clicks[file_name][N] == 0:
                    # 数据中不存在的数据 设置CTR = 0.1%
                    real_ctr = 0.001
                else:
                    real_ctr = 1.0 * \
                        datas.columns_clicks[file_name][N] / \
                        datas.columns_counts[file_name][N]

                w[N][M] = real_ctr * (1 + np.random.uniform(-1., 1.))
                v[N][M] = real_ctr * (1 + np.random.uniform(-.5, .5))


        data_sum = np.sum(datas.columns_count)
        p = np.ones(datas.count)*data_sum
        p = np.asarray(datas.columns_count, dtype=float)/p

        self.n = n
        self.m = m
        self.v = v
        self.w = w
        self.p = p


def load_Datas():
    if os.path.exists('./pickles/datas.pickle'):
        with open('./pickles/datas.pickle', "rb") as f:
            return pickle.load(f)

def save(datas):
    with open('./pickles/datas.pickle', "wb") as f:
        pickle.dump(datas, f)

if __name__ == "__main__":
    datas = Datas()
    save(datas)