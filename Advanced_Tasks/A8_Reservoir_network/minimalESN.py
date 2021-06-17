# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg 
# numpy.linalg is also an option for even fewer dependencies

# %%
# load the data
trainLen = 2000
testLen = 2000
initLen = 100
data = np.loadtxt('dataset/MackeyGlass_t17.txt')

# %%
# plot some of it
plt.figure(10).clear()
plt.plot(data[:1000])
plt.title('A sample of data')

# %%
# generate the ESN reservoir
inSize = outSize = 1 # 输入维数 K
resSize = 1000 # 储备池规模 N
a = 0.3 # leaking rate 储备池更新的速度

np.random.seed(42)

Win = (np.random.rand(resSize,1+inSize) - 0.5) * 1 # 输入矩阵 N * (1+K)
W = np.random.rand(resSize,resSize) - 0.5 # 储备池连接矩阵 N * N

# normalizing and setting spectral radius (correct, slow):
# 对W进行防缩，以满足系数的要求
# 归一化并设置谱半径
print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0])) # linalg.eig(W)[0]:特征值   linalg.eig(W)[1]:特征向量
print('done.')
W *= 1.25 / rhoW

# allocated memory for the design (collected states) matrix
X = np.zeros((1+inSize+resSize,trainLen-initLen)) # 储备池的状态矩阵X(t), 每一列是每个时刻储备池的状态
# set the corresponding target matrix directly
Yt = data[None,initLen+1:trainLen+1]  # 输出矩阵，每一行是一个时刻的输出

# run the reservoir with the data and collect X
# 输入所有的训练数据，然后得到每一时刻的输入值和储备池状态
x = np.zeros((resSize,1))
for t in range(trainLen):
    u = data[t]
    x = (1-a)*x + a*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x ) ) # vstack((1, u)) 将偏置量1加入输入序列
    if t >= initLen: # 空转100次后，开始记录储备池状态
        X[:,t-initLen] = np.vstack((1,u,x))[:,0]
    
# train the output by ridge regression
# 使用Wout根据输入值和储备池状态去拟合目标值，这是一个简单的线性回归问题。
reg = 1e-8  # regularization coefficient
# direct equations from texts:
#X_T = X.T
#Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
#    reg*np.eye(1+inSize+resSize) ) )
# using scipy.linalg.solve:
# Wout 1*1+K+N
Wout = linalg.solve( np.dot(X,X.T) + reg*np.eye(1+inSize+resSize), 
    np.dot(X,Yt.T) ).T
# linalg.inv 矩阵求逆； numpy.eye()生成对角矩阵， 规模：1+inSize+reSize, 默认对角线全1，其余全0

# run the trained ESN in a generative mode. no need to initialize here, 
# because x is initialized with training data and we continue from there.
# 使用训练数据进行前向处理得到结果
Y = np.zeros((outSize,testLen))
u = data[trainLen]
for t in range(testLen):
    x = (1-a)*x + a*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x ) )
    y = np.dot( Wout, np.vstack((1,u,x)) ) # 输出矩阵 (1*1+K+N)*此刻状态矩阵(1+K+N*1)=此刻预测值

    Y[:,t] = y # t时刻的预测值， Y: 1 * testLen

    # generative mode:
    u = y
    ## this would be a predictive mode:
    #u = data[trainLen+t+1] 

# %%
# compute MSE for the first errorLen time steps
errorLen = 500
mse = sum( np.square( data[trainLen+1:trainLen+errorLen+1] - 
    Y[0,0:errorLen] ) ) / errorLen
print('MSE = ' + str( mse ))
    
# %%
# plot some signals
# 绘制测试集的真实数据和预测数据
plt.figure(1).clear()
plt.plot( data[trainLen+1:trainLen+testLen+1], 'g' )
plt.plot( Y.T, 'b' )
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'])

# 绘制储备池中前200个时刻状态(x(t))的前20个贮备层节点值
plt.figure(2).clear()
plt.plot( X[0:20,0:200].T )
plt.title(r'Some reservoir activations $\mathbf{x}(n)$')

# 绘制输出矩阵
plt.figure(3).clear()
plt.bar( np.arange(1+inSize+resSize), Wout[0].T )
plt.title(r'Output weights $\mathbf{W}^{out}$')

plt.show()
