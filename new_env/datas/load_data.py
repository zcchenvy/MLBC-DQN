import numpy as np
import torch
from scipy.sparse import coo_matrix
import pickle
import os

DEFEAT_COLUMN = ["weekday", "hour", "IP", "region", "city", "adexchange", "domain", "slotid",
                 "slotwidth", "slotheight", "slotvisibility", "slotformat", "creative",
                 "useragent", "slotprice",
                 "advertiser"]

DEFEAT_FILES = ['1458', '2259', '2261', '2821',
                '2997', '3358', '3386', '3427', '3476']


def featTrans(name, content):
    oses = ["windows", "ios", "mac", "android", "linux"]
    browsers = ["chrome", "sogou", "maxthon",
                "safari", "firefox", "theworld", "opera", "ie"]

    content_bak = content
    content = content.lower()
    if name == "useragent":
        operation = "other"
        for o in oses:
            if o in content:
                operation = o
                break
        browser = "other"
        for b in browsers:
            if b in content:
                browser = b
                break
        return operation + "_" + browser
    if name == "slotprice":
        price = int(content)
        if price > 100:
            return "101+"
        elif price > 50:
            return "51-100"
        elif price > 10:
            return "11-50"
        elif price > 0:
            return "1-10"
        else:
            return "0"
    return content_bak


class LoadData:
    '''
    Args
    ----
    self.files/self.dsps: DSP的集合
    self.columns: 在数据中使用了哪些特征

    columns_click: 关于某个特征类型的总点击次数
    columns_count: 关于某个特征类型的总展示次数

    self.columns_clicks: 关于某个特征类型的总点击次数 -- 分类了的
    self.columns_counts: 关于某个特征类型的总展示次数 -- 分类了的
    count: 特征类型的总个数

    '''

    def __init__(self, files=DEFEAT_FILES, columns=DEFEAT_COLUMN):
        self.files = files
        self.dsps = files
        self.columns = columns

        print("制作数据，加载的特征值包含: {}".format(self.columns))
        namecol = {}
        features = {}

        max_colums = 0

        for data_file in self.files:
            file_input = open("make-ipinyou-data/" +
                                  data_file+"/train.log.txt", 'r')
            first_line = True
            for line in file_input:
                s = line.split('\t')
                if first_line:
                    first_line = False
                    for i in range(0, len(s)):
                        namecol[s[i].strip()] = i
                    continue

                content = ""

                for f in self.columns:
                    col = namecol[f]
                    content += featTrans(f, s[col])
                    content += '_'
                if content not in features:
                    features[content] = max_colums
                    max_colums += 1

        print("特征向量维度为: {}".format(max_colums))

        self.columns_click = np.zeros(max_colums)
        self.columns_count = np.zeros(max_colums)

        self.columns_clicks = {}
        self.columns_counts = {}

        for data_file in self.files:
            self.columns_clicks[data_file] = np.zeros(max_colums)
            self.columns_counts[data_file] = np.zeros(max_colums)

            file_input = open("make-ipinyou-data/" +
                                  data_file+"/train.log.txt", 'r')
            first_line = True
            for line in file_input:
                s = line.split('\t')

                if first_line:
                    first_line = False
                    continue

                content = ""

                for f in self.columns:
                    col = namecol[f]
                    content += featTrans(f, s[col])
                    content += '_'

                self.columns_count[features[content]] += 1
                self.columns_counts[data_file][features[content]] += 1

                if s[0] == "1":
                    self.columns_click[features[content]] += 1
                    self.columns_clicks[data_file][features[content]] += 1
        self.count = max_colums

        with open('./pickles/{}-{}.pickle'.format(str(self.files), str(self.columns)), "wb") as f:
            pickle.dump(self, f)


def load_LoadData(files=DEFEAT_FILES, columns=DEFEAT_COLUMN):
    if os.path.exists('./pickles/{}-{}.pickle'.format(str(files), str(columns))):
        with open('./pickles/{}-{}.pickle'.format(str(files), str(columns)), "rb") as f:
            return pickle.load(f)
    else:
        return LoadData(files=files, columns=columns)


def save(datas):
    with open('./pickles/{}-{}.pickle'.format(str(datas.files), str(datas.columns)), "wb") as f:
        pickle.dump(datas, f)
