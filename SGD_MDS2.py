import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from math import sqrt
import itertools

from numba import jit

import math
import random

norm = lambda x: np.linalg.norm(x,ord=2)


class SGD:
    def __init__(self,dissimilarities,k=5,weighted=True,w = np.array([]), init_pos=np.array([])):
        self.d = dissimilarities
        self.d_max = np.max(dissimilarities)
        self.d_min = 1
        self.n = len(self.d)
        if init_pos.any():
            self.X = np.asarray(init_pos)
        else: #Random point in the chosen geometry
            self.X = np.zeros((len(self.d),2))
            for i in range(len(self.d)):
                self.X[i] = self.init_point()
            self.X = np.asarray(self.X)

        a = 1
        b = 1
        if weighted:
            # self.w = set_w(self.d,k)
            self.w = w
        else:
            self.w = np.array([[ 1 if self.d[i][j] == 1 else 0 for i in range(self.n)]
                        for j in range(self.n)])

        w_min = 1/pow(self.d_max,2)
        self.w_max = 1/pow(self.d_min,2)
        self.eta_max = 1/w_min
        epsilon = 0.1
        self.eta_min = epsilon/self.w_max

    def get_sched(self,num_iter):
        lamb = np.log(self.eta_min/self.eta_max)/(num_iter-1)
        sched = lambda count: self.eta_max*np.exp(lamb*count)
        #sched = lambda count: 1/np.sqrt(count+1)
        return np.array([sched(count) for count in range(100)])

    def solve(self,num_iter=1500,debug=False,t=1,radius=False, k=1,tol=1e-8):
        import autograd.numpy as np
        from autograd import grad
        from sklearn.metrics import pairwise_distances
        import itertools

        indices = np.array(list(itertools.combinations(range(self.n), 2)))

        d = self.d
        w = 0.5* np.copy(self.w) if not radius else 0.5*np.copy((self.d <= k).astype('int'))
        X = self.X
        N = len(X)

        if debug:
            hist = [np.ones(X.shape) for count in range(num_iter+1)]

        sizes = np.zeros(num_iter+1)
        movement = lambda X,X_c,step: np.sum(np.sum(X_c ** 2, axis=1) ** 0.5) / (N * step * np.max(np.max(X, axis=0) - np.min(X, axis=0)))

        eps = 1e-13
        epsilon = np.ones(d.shape)*eps
        def pair_stress(v,u,t,dij,wij):                 # Define a function
            stress, l_sum = 0, 1+t

            mag = np.linalg.norm(v-u)
            return (1/l_sum) * (wij * np.square(mag-dij)) - (t/l_sum) * np.log(mag)

        step,change,momentum = 0.001, 0.0, 0.5
        grad_stress = grad(pair_stress)
        cost = 0

        #t = 0.6
        for epoch in range(num_iter+1):
            step = self.compute_step_size(epoch,num_iter)
            step = step if step < 1 else 1

            for i,j in indices:
                change = grad_stress(X[i],X[j],t,d[i][j],w[i][j])
                X[i] -= step * change
                X[j] += step * change

            # x_prime = grad_stress(X,t)
            #
            # new_change = step * x_prime + momentum * change
            #
            # X -= new_change
            #
            # if abs(new_change-change).max() < 1e-3: momentum = 0.8
            # sizes[epoch] = movement(X,new_change,step)
            #
            # change = new_change

            if epoch > -1:
                #max_change = sizes[epoch - 40 : epoch].max()
                #cost = stress(X,t)
                print("Epoch: {} . Cost value: {} . ".format(epoch,int(cost)),end='\r')
                #if max_change < tol: break
                # if epoch % 101 == 0:
                #     if stress(X,t) < -70000: break

            #print(stress(X,t))
            #self.X = X
            if debug:
                hist[epoch] = X.copy()
        self.X = X
        print()
        if debug:
            return hist
        return X.copy()



    def compute_step_size(self,count,num_iter):
        #return 1/pow(5+count,1)

        lamb = math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return self.eta_max*math.exp(lamb*count)


    def init_point(self):
        return [random.uniform(-1,1),random.uniform(-1,1)]


def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product




def scale_matrix(d,new_max):
    d_new = np.zeros(d.shape)

    t_min,r_min,r_max,t_max = 0,0,np.max(d),new_max

    for i in range(d.shape[0]):
        for j in range(i):
            m = ((d[i][j]-r_min)/(r_max-r_min))*(t_max-t_min)+t_min
            d_new[i][j] = m
            d_new[j][i] = m
    return d_new


def set_w(d,k):
    f = np.zeros(d.shape)
    for i in range(len(d)):
        for j in range(len(d)):
            if i == j:
                f[i][j] = 100000
            else:
                f[i][j] = d[i][j]
    f += np.random.normal(scale=0.1,size=d.shape)
    k_nearest = [get_k_nearest(f[i],k) for i in range(len(d))]

    #1/(10*math.exp(d[i][j]))
    w = np.asarray([[ 1e-5 if i != j else 0 for i in range(len(d))] for j in range(len(d))])
    for i in range(len(d)):
        for j in k_nearest[i]:
            if i != j:
                w[i][j] = 1
                w[j][i] = 1

    return w
