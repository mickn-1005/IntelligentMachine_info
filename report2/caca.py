import numpy as np
import matplotlib.pyplot as plt

# ジニ不純度を計算する関数
def gini(p):
    return 1-(p**2).sum()

# ノードのクラス
class Node:
    def __init__(self,X,z,level):
        self.level = level
        self.n = len(z)
        self.branch = []
        if(len(np.unique(z))==1):
            self.z = z[0]
        elif(level==0):
            self.z = np.bincount(z).argmax()
        else:
            self.gn = 1
            for j in range(X.shape[1]):
                x = X[:,j]
                xas = x.argsort()
                x_sort = x[xas]
                z_sort = z[xas]
                l = (x_sort[1:]+x_sort[:-1])/2
                l = l[z_sort[1:]!=z_sort[:-1]]
                for thres in l:
                    split = thres>x
                    z_left = z[split]
                    z_right = z[~split]
                    n_left = float(len(z_left))
                    n_right = float(len(z_right))
                    gn = (gini(np.bincount(z_left)/n_left)*n_left+gini(np.bincount(z_right)/n_right)*n_right)/self.n
                    if(self.gn>gn):
                        self.gn = gn
                        self.j = j
                        self.thres = thres
            o = (self.thres>X[:,self.j])
            self.branch = [Node(X[o],z[o],level-1),Node(X[~o],z[~o],level-1)]

    def __call__(self,X):
        if(self.branch==[]):
            return self.z
        else:
            o = self.thres>X[:,self.j]
            return np.where(o,self.branch[0](X),self.branch[1](X))

# 決定木のクラス
class DecisionTree:
    def __init__(self,depth):
        self.depth = depth # 分割する最大の数

    def train(self,X,z):
        self.ne = Node(X,z,self.depth) # 根からはじめ、木の中のノードを作る

    def predict(self,X): # 予測
        return self.ne(X)

class RandomForest:
    def __init__(self,depth,n_tree=10):
        self.depth = depth
        self.n_tree = n_tree # 決定木の数

    def train(self,X,z):
        n = len(z)
        self.k = z.max()+1
        self.tree = []
        for i in range(self.n_tree):
            tt = DecisionTree(self.depth)
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

def decision_tree(X, z):
    dt = DecisionTree(100)
    dt.train(X[:4000],z[:4000])

    return dt

def random_forest(X, z, size=500):
    ranfo = RandomForest(100, size)
    ranfo.train(X[:4000],z[:4000])

    return ranfo

def plots(clf, X, z):
    nmesh = 200
    mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),nmesh),np.linspace(X[:,1].min(),X[:,1].max(),nmesh))
    mX = np.stack([mx.ravel(),my.ravel()],1)
    mz = clf.predict(mX).reshape(nmesh,nmesh)
    plt.gca(aspect=1,xlim=[mx.min(),mx.max()],ylim=[my.min(),my.max()])
    plt.contourf(mx,my,mz,alpha=0.4,cmap='rainbow',zorder=0)
    plt.show()
    plt.gca(aspect=1,xlim=[mx.min(),mx.max()],ylim=[my.min(),my.max()])
    plt.scatter(X[4000:5000,0],X[4000:5000,1],alpha=0.6,c=z[4000:5000],edgecolor='k',cmap='rainbow')
    plt.contourf(mx,my,mz,alpha=0.4,cmap='rainbow',zorder=0)
    plt.show()

def accuracy(clf, X, z):
    pred = clf.predict(X[4000:5000])
    return np.sum(pred==z[4000:5000])/1000

from sklearn import datasets

X,z = datasets.make_blobs(n_samples=5000,n_features=2,centers=5,cluster_std=1.7,random_state=2)
dt = decision_tree(X,z)
#rf1 = random_forest(X,z,10)
#rf2 = random_forest(X,z,100)
rf3 = random_forest(X,z)
#rf4 = random_forest(X,z,1000)
plots(dt, X, z)
#plots(rf1, X, z)
#plots(rf2, X, z)
plots(rf3, X, z)
#plots(rf4, X, z)

print('acc of decision tree:', accuracy(dt, X,z))
#print('acc of random forest1:', accuracy(rf1, X,z))
#print('acc of random forest2:', accuracy(rf2, X,z))
print('acc of random forest3:', accuracy(rf3, X,z))
#print('acc of random forest4:', accuracy(rf4, X,z))
