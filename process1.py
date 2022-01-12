import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf

X=np.empty((100,2))
X2=np.random.random(size=(100,2))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.6 * X[:,0] + 3. + np.random.normal(0., 10., size=100)
result=np.dot(X.flatten(),X2.flatten())


def demean(X):
    # axis=0按列计算均值，即每个属性的均值，1则是计算行的均值
    return (X - np.mean(X, axis=0))


X_demean = demean(X)
# 注意看数据分布没变，但是坐标已经以原点为中心了

# random dithering 2-norm
def random_dithering(x, s):
    if s == 0:
        return np.zeros(x.shape)
    norm = np.linalg.norm(x, axis=0)
    noise = np.random.uniform(0, 1, x.shape)
    floored = np.floor(s * np.abs(x) / norm + noise)
    compressed = norm / s * np.sign(x) * floored
    return compressed

def f(w,X):
    return np.sum((X.dot(w)**2))/len(X)
def df_math(w,X):
    return X.T.dot(X.dot(w))*2./len(X)
# 验证梯度求解是否正确，使用梯度调试方法：
def df_debug(w, X, epsilon=0.0001):
    # 先创建一个与参数组等长的向量
    res = np.empty(len(w))
    # 对于每个梯度，求值
    for i in range(len(w)):
        w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X) - f(w_2, X)) / (2 * epsilon)
    return res
def direction(w):
    return w / np.linalg.norm(w)

# 梯度上升法代码
def gradient_ascent(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    w = direction(initial_w)
    cur_iter = 0
    while cur_iter < n_iters:
        gradient = df_math(w,X)
        last_w = w
        w = last_w + eta * gradient
        w = direction(w)    # 将w转换成单位向量
        if (abs(f(w,X) - f(last_w,X)) < epsilon):
            print("上升{}次后到达极值".format(cur_iter))
            break
        cur_iter += 1
    print("meiyong")
    return w

initial_w = np.random.random(X.shape[1])
eta = 0.001

#w = gradient_ascent(df_debug, X_demean, initial_w, eta)
w = gradient_ascent(df_math, X_demean, initial_w, eta)
t=np.empty(shape=(1, X.shape[1]))
# 输出
t[0]=w
np.array([0.76567331, 0.64322965])


X_new = X - X.dot(w).reshape(-1,1) * w
X_T=X_new.dot(t.T)##点乘一维必须放在右边，就是作为行，根一百行进行点积。相乘的话必须一维数组的数等于二维数组的列。
print(X.dot(w).reshape(-1,1))
print(np.dot(X_new.flatten(),X.flatten()))
w_new = gradient_ascent(df_math, X_new, initial_w, eta)

print(w_new)
# 输出：

np.array([-0.64320916,  0.76569052])
