
import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import _pickle as cPickle

class PCA:

    def __init__(self, n_component):
        assert n_component >= 1, 'n_component is invalidate'
        self.n_component = n_component
        self.components_ = None

    def __repr__(self):
        return 'PCA(n_component=%d)' % self.n_component

    def fit(self,X, eta, n_iter=1000, epsilon=0.0001):
        '''
        主成分分析
        :param X:
        :param eta:
        :param n_iter:
        :param epsilon:
        :return:
        '''
        assert X.shape[1] >= self.n_component, 'X is invalidate'

        '''均值归零'''
        def demean(X):
            return X - np.mean(X, axis=0)

        '''方差函数'''
        def f(w, X):
            return np.sum(X.dot(w)**2) / len(X)

        '''方差函数导数'''
        def df_ascent(w, X):
            return X.T.dot(X.dot(w)) * 2 / len(X)

        '''将向量化简为单位向量'''
        def direction(w):
            return w / np.linalg.norm(w)

        '''寻找第一主成分'''
        def first_component(w, X, eta, n_iter=1000, epsilon=0.0001):
            i_iter = 0
            while i_iter < n_iter:
                last_w = w
                gradient = df_ascent(w, X)
                w += eta * gradient
                w = direction(w)
                if abs(f(w, X) - f(last_w, X)) < epsilon:
                    break
                i_iter += 1
            return w

        self.components_ = np.empty(shape=(self.n_component, X.shape[1]))
        X = demean(X)
        for i in range(self.n_component):
            w = np.random.random(X.shape[1])

            w = first_component(w, X, eta, n_iter, epsilon)
            X = X - (X.dot(w)).reshape(-1, 1) * w
            self.components_[i, :] = w
        return self

    def transform(self, X):
        '''
        将X映射到各个主成分中
        :param X:
        :return:
        '''
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        '''
        将低维数据转回高维
        :param X:
        :return:
        '''
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

def demean(X):
    # axis=0按列计算均值，即每个属性的均值，1则是计算行的均值
    return (X - np.mean(X, axis=0))
# # 注意看数据分布没变，但是坐标已经以原点为中心了
# def f(w, X):
#     return np.sum((X.dot(w) ** 2)) / len(X)
# def df_math(w, X):
#     return X.T.dot(X.dot(w)) * 2. / len(X)
# # 验证梯度求解是否正确，使用梯度调试方法：
# def df_debug(w, X, epsilon=0.0001):
#     # 先创建一个与参数组等长的向量
#     res = np.empty(len(w))
#     # 对于每个梯度，求值
#     for i in range(len(w)):
#         w_1 = w.copy()
#         w_1[i] += epsilon
#         w_2 = w.copy()
#         w_2[i] -= epsilon
#         res[i] = (f(w_1, X) - f(w_2, X)) / (2 * epsilon)
#     return res
# def direction(w):
#     return w / np.linalg.norm(w)
# 梯度上升法代码
# def gradient_ascent(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
# def gradient_ascent(df, X, initial_w, eta, n_iters=10000, epsilon=1e-8):
#     w = direction(initial_w)
#     cur_iter = 0
#     while cur_iter < n_iters:
#         gradient = df(w, X)
#         last_w = w
#         w = w + eta * gradient
#         w = direction(w)  # 将w转换成单位向量
#         print(abs(f(w, X) - f(last_w, X)))
#         if (abs(f(w, X) - f(last_w, X)) < epsilon):
#             print("上升{}次后到达极值".format(cur_iter))
#             break
#         cur_iter += 1
#     print("未到达极值")
#     return
def compress_pkl(yasuo,xiangliang,compress,size):
    device = torch.device("cuda")

    with open(yasuo,"rb") as f1:
        # with open("2_inverse.pkl","wb") as f2:
        with open(xiangliang, "wb") as f2:
            with open(compress,"wb") as f3:
                for i in range(size):
                    print(i)
                    u=cPickle.load(f1)


                    list1=[]
                    list=[]
                    t=0
                    for x in u:
                        X = x.cpu().numpy()
                        X_demean = demean(X)
                        if X.ndim == 1:
                            list.append(x)
                            list1.append(torch.from_numpy(torch.ones(x.shape).float().numpy()))
                            continue
                        # initial_w = np.random.random(X.shape[1])
                        # initial_w = np.array(initial_w, dtype=np.float32)
                        # eta = 0.01
                        # 使用梯度上升求出 w 方向，使得样本投影到w 方向后 样本 方差最大
                        # t += 1
                        # if t == 2:
                        #     w2 = gradient_ascent(df_debug, X_demean, initial_w, eta)
                        # w = gradient_ascent(df_math, X_demean, initial_w, eta)
                        w_test = PCA(1)
                        # if i==0:
                        #     w_test.fit(X,0.01,100)
                        # else:
                        w_test.fit(X, 0.01)
                        X_new = w_test.transform(X)
                        X_new = np.array(X_new, dtype=np.float32)
                        #X_inverse =w_test.inverse_transform(X_new)
                        w_tensor = torch.from_numpy(X_new).to(device)
                        xl=torch.from_numpy(w_test.components_).to(device)
                        #w_tensor1 = torch.from_numpy(X_inverse).to(device)

                        list.append(w_tensor)
                        list1.append(xl)
                    cPickle.dump(list, f3)
                    cPickle.dump(list1,f2)

            # X=torch.from_numpy(u)
            # t = tf.shape(u.cpu())
            # X[:,0] = np.random.uniform(0., 100., size=100)
            # X[:,1] = 0.75 * X[:,0] + 3. + np.random.normal(0., 10., size=100)
            #

            # plt.scatter(X[:,0],X[:,1])
            # plt.show()

                # 输出
                # plt.scatter(X_demean[:,0],X_demean[:,1])
                # plt.plot([0,w[0]*30],[0,w[1]*30], color='red')
                # plt.show()
                #xx-new chuizhi yu w
            # X_new = X - X.dot(w).reshape(-1,1) * w
            # X_new2=torch.from_numpy(X_new).to(device)
                # plt.scatter(X_new[:,0], X_new[:,1])
                # plt.imshow(X_new)
                # plt.show()
            # w_new = gradient_ascent(df_math, X_new, initial_w, eta)
            # w_tensor=torch.from_numpy(w_new).to(device)
                # 输出：

            # def first_n_component(n, X, eta=0.001, n_iters=1e4, epsilon=1e-8):
            #     X_pca = X.copy()
            #     X_pca = demean(X_pca)
            #     res = []
            #     for i in range(n):
            #         initial_w = np.random.random(X_pca.shape[1])
            #         w = gradient_ascent(df_math, X_pca, initial_w, eta)
            #         res.append(w)
            #         X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
            #     return res
