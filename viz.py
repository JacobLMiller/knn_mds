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


norm = np.linalg.norm
import itertools
#@jit(nopython=True,cache=True)
def full_norm(n):
    shuffle = np.random.shuffle
    randint = random.randint
    indices = list(itertools.combinations(range(n), 2))

    X = np.random.uniform(0,1,(n,2))
    d = np.random.uniform(0,1,(n,n))
    w = np.ones(d.shape)
    step = 0.001

    for epoch in range(30):

        change = 10

        for i,j in indices:

            pq = X[i]-X[j]
            mag = norm(pq)
            mag_grad = pq/mag


            mu = step*w[i][j]
            if mu >= 1: mu = 1

            r = (mu*(mag-d[i][j]))/(2*mag)
            stress = r*pq

            mu1 = step if step < 1 else 1
            repulsion = -((step/mag) * mag_grad)

            l_sum = 1+0.01
            m = (1/l_sum)*stress + (0.01/l_sum)*repulsion

            X[i] -= m
            X[j] += m

        shuffle(indices)

from math import sqrt
#@jit(nopython=True,cache=True)
def full_custom(n):

    shuffle = np.random.shuffle
    randint = random.randint
    indices = list(itertools.combinations(range(n), 2))

    X = np.random.uniform(0,1,(n,2))
    d = np.random.uniform(0,1,(n,n))
    w = np.ones(d.shape)
    step = 0.001
    for epoch in range(30):

        change = 10

        for i,j in indices:

            #old = np.linalg.norm(X[i]-X[j])
            #pq = X[i]-X[j]
            #square = pq ** 2
            mag = ( (X[i][0]-X[j][0])**2 + (X[i][1]-X[j][1])**2) ** 0.5
            #mag = norm(pq)
            #mag_grad = pq/mag
            dx = (X[i][0]-X[j][0])/mag
            dy = (X[i][1]-X[j][1])/mag

            #w = 1/(dij**2)
            mu = step*w[i][j]
            if mu >= 1: mu = 1

            r = (mu*(mag-d[i][j]))/(2*mag)
            stressx = r*dx
            stressy = r*dy

            mu1 = step if step < 1 else 1
            repulsionx = -((step/mag) * dx)
            repulsiony = -((step/mag) * dy)

            l_sum = 1+0.01
            mx = (1/l_sum)*stressx + (0.01/l_sum)*repulsionx
            my = (1/l_sum)*stressy + (0.01/l_sum)*repulsiony

            X[i][0] -= mx
            X[i][1] -= my
            X[j][0] += mx
            X[j][1] += my

        shuffle(indices)




def timing(n):
    setup = '''
import numpy as np
from __main__ import full_custom
    '''

    loop = '''
full_custom(n={})
    '''.format(n)

    number = 10
    import timeit
    hit = timeit.timeit(setup=setup,
                        stmt=loop,number=number)
    print(hit/number)
    return hit/number

def timing2(n):
    setup = '''
import numpy as np
from __main__ import full_norm
    '''

    loop = '''
full_norm(n={})
    '''.format(n)

    number = 100
    import timeit
    hit = timeit.timeit(setup=setup,
                        stmt=loop,number=number)
    print(hit/number)
    return hit/number

if __name__ == "__main__":
    toplot = [timing(n) for n in range(10,1000,100)]
    toplot2 = [timing2(n) for n in range(10,1000,100)]
    import matplotlib.pyplot as plt
    plt.suptitle("with jit")
    plt.plot(list(range(10,1000,100)), toplot,label='custom')
    plt.plot(list(range(10,1000,100)), toplot2,label='numpy')
    plt.legend()
    plt.show()
