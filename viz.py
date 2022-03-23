import cProfile
import Test2
#cProfile.run('Test2.example()')
import numpy as np
import timeit


n = 5

u,v = np.random.uniform(0,1,2), np.random.uniform(0,1,2)
nonumpy = lambda: (sum( ((u[0]-v[0])**2, (u[1]-v[1])**2) )) ** 0.5

import itertools
indices = list(itertools.combinations(range(n), 2))
lis = [i for i in range(len(indices))]

print(type(indices))
print(len(indices))

norm = np.linalg.norm


from numba import jit

@jit(nopython=True)
def np_jit(n):
    x = np.random.uniform(0,1,(n,2))

    i,j = 3,4
    for _ in range(30):
        y = norm(x[i]-x[j])


def np_norm(n):
    #d = np.random.uniform(0,1,(n,n))
    x = np.random.uniform(0,1,(n,2))

    i,j = 3,4
    for _ in range(30):
        y = norm(x[i]-x[j])

def np_norm_loaded(n):
    #d = np.random.uniform(0,1,(n,n))
    x = np.random.uniform(0,1,(n,2))
    i,j = 3,4
    for _ in range(30):
        y = norm(x[i]-x[j])

def custom(n):
    x = [ [random.random(),random.random()] for _ in range(n)]
    i,j = 3,4
    for _ in range(30):
        y = (sum( ((x[i][0]-x[j][0])**2, (x[i][1]-x[j][1])**2) )) ** 0.5


import random
def custom2():
    x = [random.random() for _ in range(2*n)]
    i,j = 1,2
    for _ in range(10):
        y = (sum( ((x[2*i]-x[2*j])**2, (x[2*i+1]-x[2*j+1])**2) )) ** 0.5


def full_norm(n):
    import itertools
    indices = list(itertools.combinations(range(n), 2))
    shuffle = np.random.shuffle

    X = np.random.uniform(0,1,(n,2))
    d = np.random.uniform(0,1,(n,n))
    w = np.ones(d.shape)
    step = 0.001
    for epoch in range(30):

        change = 10

        for i,j in indices:
            #old = np.linalg.norm(X[i]-X[j])
            pq = X[i]-X[j]
            #mag = (pq[0]*pq[0] + pq[1]*pq[1]) ** 0.5
            mag = norm(pq)
            mag_grad = pq/mag

            #w = 1/(dij**2)
            mu = step*w[i][j]
            if mu >= 1: mu = 1

            r = (mu*(mag-d[i][j]))/(2*mag)
            stress = r*pq

            mu1 = step if step < 1 else 1
            repulsion = -((step/mag) * mag_grad)
            # if mag > 3*diam:
            #     repulsion *= 1
            #
            l_sum = 1+0.01
            m = (1/l_sum)*stress + (0.01/l_sum)*repulsion

            X[i] -= m
            X[j] += m

        #shuffle(indices)

import time
sizes = list(range(5,100,10))
np_m = np.zeros(len(sizes))
cust = np.zeros(len(sizes))

iter = 1

for n in range(len(sizes)):
    for i in range(iter):
        start = time.perf_counter()
        full_norm(sizes[n])
        end = time.perf_counter()
        np_m[n] += end-start

        start = time.perf_counter()
        np_norm(sizes[n])
        end = time.perf_counter()
        cust[n] += end-start

cust /= iter
np_m /= iter


import matplotlib.pyplot as plt
plt.plot(sizes[1:],np_m[1:],label='numpy')
plt.plot(sizes[1:],cust[1:],label='custom')
plt.legend()
plt.show()

import cProfile
cProfile.run('full_norm(25)')
