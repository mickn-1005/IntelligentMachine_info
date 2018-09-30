# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
決定木のMNIST実装
@author: mickn
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
# クロスエントロピー計算関数（コスト関数）
def cross_entropy(p):
    return -np.sum(p * np.log(p))

class Node_tree():
    """docstring for Node."""
    def __init__(self, x_train, y_train, level):
        print('level : ', level)
        self.level = level
        self.size = len(y_train)    # ノード内のデータサイズ
        self.branch = []    # このノードから生えている枝のリスト
        if len(np.unique(y_train)) in [0,1]:  # 枝が一本のみ
            print('no data included')
            self.y = y_train
        elif level==0:
            self.y = np.bincount(y_train).argmax()
        else:
            self.cen = 10    # コスト関数初期化
            for j in range(x_train.shape[1]):
                x = x_train[:,j]
                xas = x.argsort()
                x_sort = x[xas]
                y_sort = y_train[xas]
                l = (x_sort[1:]+x_sort[:-1])/2
                l = l[y_sort[1:]!=y_sort[:-1]]
                for thres in l:
                    split = (thres>x)
                    y1 = y_train[split]
                    y2 = y_train[~split]
                    size1 = float(len(y1))
                    size2 = float(len(y2))
                    #print(size1, "---", size2)
                    # cen = cross_entropy(np.bincount(y1)/size1)
                    cen = (cross_entropy(np.bincount(y1)/size1)*size1+cross_entropy(np.bincount(y2)/size2)*size2)/self.size
                    
                    if(self.cen>cen):
                        self.cen = cen
                        self.j = j
                        self.thres = thres
                        print(self.thres, '---', self.cen)
            print('length:',len(x_train),len(y_train))
            out = (self.thres > x_train[:,self.j])
            self.branch = [Node_tree(x_train[out], y_train[out], level-1), Node_tree(x_train[~out], y_train[~out], level-1)]

    def __call__(self, X):
        if self.branch==[]:
            return self.y
        else:
            o = self.thres>X[:,self.j]
            return np.where(o, self.branch[0](X), self.branch[1](X))

class dec_tree():
    """docstring for cart."""
    def __init__(self, depth):
        self.depth = depth

    def train(self, x_train, y_train):
        self.root = Node_tree(x_train, y_train, self.depth)

    def predict(self, x_test):
        return self.root(x_test)



class rand_for():
    def __init__(self,depth,n_tree=10):
        self.depth = depth
        self.n_tree = n_tree

    def train(self,X,z):
        n = len(z)
        self.k = z.max()+1
        self.tree = []
        for i in range(self.n_tree):
            tt = dec_tree(self.depth)
            s = np.random.choice(n,n)
            tt.train(X[s],z[s])
            self.tree.append(tt)

    def _predict(self,X):
        result = 0
        for tt in self.tree:
            result += tt.predict(X)[:,None]==np.arange(self.k)
        return result

    def predict(self,X): # 答えを出す
        return self._predict(X).argmax(1)

    def predict_proba(self,X): # 可能性を出す
        return self._predict(X)/self.n_tree

(x_train, y_train),(x_test,y_test) = mnist.load_data()
dt = dec_tree(12)
x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)
dt.train(x_train[:500],y_train[:500])

#def evaluation(pred,y_te):
    
