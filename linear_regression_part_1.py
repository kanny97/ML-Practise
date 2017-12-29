import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.getcwd() + '\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())  # data head printed to check
# add a column of ones for vectorization
data.insert(0, 'ones', 1)
print(data)
cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:]
print(X)
print(y)

def computeCost(X,y,theta):
    m=X.shape[0]
    h=X*theta.T
    temp=np.power(h-y,2)
    cost=sum(temp)/(2*m)
    return cost

def gradientDescent(X,y,theta,alpha,iters):
    cost=np.zeros(iters)
    parameters=int(theta.ravel().shape[1])
    for i in range(iters):
        m=len(X)
        err=X*theta.T-y
        theta=theta- (alpha / m) * (err.T * X)  #vectorization step
        cost[i]=computeCost(X,y,theta)
    return theta,cost

X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))
print('The cost for initial theta parameters is:',computeCost(X,y,theta))
alpha=0.01
iters=1500
g,cost=gradientDescent(X,y,theta,alpha,iters)
print(g)
