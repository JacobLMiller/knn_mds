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

@jit(nopython=True)
def satisfy2(v,u,di,we,step,t=1,count=0):

    l_sum = 1+t

    m = np.zeros(v.shape)

    l_sum = 1+t


    if di <= 1:
        wc = step / pow(di,2)

        pq = v-u #Vector between points

        mag = np.linalg.norm(pq)
        #r = (mag-self.d[i][j])/2 #min distance to satisfy constraint
        wc = step

        # if we < 0.9 and random.random()<0.1:
        #     r = (mag-10*self.d_max)/2
        #     wc = step
        # elif we > 0.9:
        r = (mag-(di))/2

        if wc > 1:
            wc = 1
        r = wc*r
        m += (1/l_sum)*((pq*r) /mag)
        #m *= t

        #return v-m, u+m
    #else:
    wc = step if step < 0.1 else 0.1
    pq = v-u
    mag = np.linalg.norm(pq)
    if wc > 0.1:
        wc = 0.1
    r = pq/(mag ** 2)
    r *= wc
    m += -(t/l_sum)*r
    return v-m,u+m

@jit(nopython=True)
def satisfy(v,u,di,we,step,N,t=1):

    we = 1 if di == 1 else 0

    l_sum = 1+t

    m = np.zeros(v.shape)

    wc = 1

    pq = v-u
    mag = np.linalg.norm(pq)

    stress = (mag-di) * (pq/mag)
    stress *= we*step


    repulsion = -t*(pq/(mag **2)) * (1-we)

    m = (1/l_sum) * stress + (t/l_sum) * repulsion

    return v-m, u+m

@jit(nopython=True)
def old_satisfy(v,u,di,we,step,t=1,count=0,max_change=0,mom=0):

    wc = step
    pq = v-u #Vector between points
    #mag = geodesic(self.X[i],self.X[j])
    mag = np.linalg.norm(pq)
    #r = (mag-self.d[i][j])/2 #min distance to satisfy constraint
    wc = (1/pow(di,2))*step

    # if we < 0.9 and random.random()<0.1:
    #     r = (mag-10*self.d_max)/2
    #     wc = step
    # elif we > 0.9:
    r = (mag-(di))/2

    if wc > 1:
        wc = 1
    r = wc*r
    m = (pq*r) /mag


    #m = 0.8 * mom + 0.2*m


    new_mag = np.linalg.norm((v-m)-(u+m))
    max_change = max(max_change,abs(mag-new_mag))

    return v-m, u+m, max_change

@jit(nopython=True)
def get_stress(X,d):
    n, stress = len(X),0
    for i in range(n):
        for j in range(i):
            stress += pow(d[i][j] - np.linalg.norm(X[i]-X[j]),2) / pow(d[i][j],2)

    return stress

@jit(nopython=True)
def solve(X,w,d,schedule,indices,num_iter=15,epsilon=1e-3,debug=False,t=1):
    step = 400
    shuffle = random.shuffle
    shuffle(indices)
    max_change=0


    for count in range(num_iter):
        max_change = 0
        for i,j in indices:
            X[i],X[j] = satisfy(X[i],X[j],d[i][j],w[i][j],step,t=t,count=count)

        step = schedule[min(count,len(schedule)-1)]
        #t = 0 if count < 10 else 0.6
        shuffle(indices)


    return X



@jit(nopython=True)
def calc_cost(X,d,w,t):
    cost, norm, n, l_sum = 0, np.linalg.norm, len(X), 1+t
    for i in range(n):
        for j in range(i):
            pq = X[i]-X[j]
            mag = norm(pq)
            near = pow(mag-d[i][j],2) if w[i][j] >= 1 else 0
            far = -np.log(mag) if w[i][j] < 1 else 0
            cost += (1/l_sum) * near + (t/l_sum) * far
    return cost

@jit(nopython=True)
def debug_solve(X,w,d,schedule,indices,num_iter=15,epsilon=1e-3,debug=False,t=1):
    step = 1
    shuffle = random.shuffle
    shuffle(indices)
    max_change = 0
    n = len(X)
    #schedule = np.array([1/(np.sqrt(count+10)) for count in range(num_iter)])


    diam = np.max(d)
    indiam = 1/diam
    prev_cost = 80000

    for count in range(num_iter):
        t = (1)/(count + 1)
        t = 0.6
        for _ in range(20):
            max_change = 0
            for i,j in indices:
                we = w[i][j] if w[i][j] == 1 else w[j][i]
                before = np.linalg.norm(X[i]-X[j])
                X[i],X[j] = satisfy(X[i],X[j],d[i][j],we,step,N=n,t=t)
                after = np.linalg.norm(X[i]-X[j])
                max_change = max(max_change, abs(after-before))
            if max_change < 1e-5:
                break

        step = schedule[min(count,len(schedule)-1)]
        step = 0.1

        shuffle(indices)
        cost = calc_cost(X,d,w,t)
        print(cost)

        if abs(prev_cost-cost) < epsilon:
            break
        prev_cost = cost


        yield X.copy()

    yield X.copy()
    return X


class SGD_d:
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

    def solve(self,num_iter=15,debug=False,t=1,radius=False, k=1,tol=1e-8):
        import autograd.numpy as np
        from autograd import grad
        from sklearn.metrics import pairwise_distances

        d = self.d
        w = 0.5* np.copy(self.w) if not radius else 0.5*np.copy((self.d <= k).astype('int'))
        X = self.X
        N = len(X)

        sizes = np.zeros(num_iter)
        movement = lambda X,X_c,step: np.sum(np.sum(X_c ** 2, axis=1) ** 0.5) / (N * step * np.max(np.max(X, axis=0) - np.min(X, axis=0)))

        eps = 1e-13
        epsilon = np.ones(d.shape)*eps
        def stress(X,t):                 # Define a function
            stress, l_sum = 0, 1+t


            #Stress
            ss = (X * X).sum(axis=1)
            diff = ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * np.dot(X,X.T)
            diff = np.sqrt(np.maximum(diff,epsilon))
            stress = np.sum( w * np.square(d-diff) )

            #repulsion
            r = -np.sum( np.log(diff+eps) )

            return (1/l_sum) * np.sum(stress) + (t/l_sum) * r

        step,change,momentum = 0.001, 0.0, 0.5
        grad_stress = grad(stress)
        print(stress(X,t))
        t = 0.1
        for epoch in range(num_iter):

            x_prime = grad_stress(X,t)

            new_change = step * x_prime + momentum * change

            X -= new_change

            if abs(new_change-change).max() < 1e-3: momentum = 0.8
            sizes[epoch] = movement(X,new_change,step)

            change = new_change

            if epoch > 40:
                max_change = sizes[epoch - 40 : epoch].max()
                print("Max change over last 40 epochs: {}".format(max_change),end='\r')
                if max_change < tol: break

            #print(stress(X,t))
            self.X = X
            yield X.copy()
        self.X = X
        return X.copy()


    def calc_distortion(self,X,d):
        distortion = 0
        for i in range(self.n):
            for j in range(i):
                distortion += abs((norm(X[i]-X[j])-d[i][j]))/d[i][j]
        return (1/choose(self.n,2))*distortion

    def calc_gradient(self,i,j):
        X0 = tf.Variable(self.X)
        with tf.GradientTape() as tape:
            Y = self.calc_stress(X0)
        dy_dx = tape.gradient(Y,X0).numpy()
        #dy = dy_dx.numpy()
        for i in range(len(self.d)):
            dy_dx[i] = normalize(dy_dx[i])
        return dy_dx

    def compute_step_size(self,count,num_iter):
        return 1/pow(5+count,1)

        lamb = math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return self.eta_max*math.exp(lamb*count)


    def init_point(self):
        return [random.uniform(-1,1),random.uniform(-1,1)]


def normalize(v):
    mag = pow(v[0]*v[0]+v[1]*v[1],0.5)
    return v/mag


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

def get_neighborhood(X,d,rg = 1):
    """
    How well do the local neighborhoods represent the theoretical neighborhoods?
    Closer to 1 is better.
    Measure of percision: ratio of true positives to true positives+false positives
    """
    def get_k_embedded(X,k_t):
        dist_mat = [[norm(X[i]-X[j]) if i != j else 10000 for j in range(len(X))] for i in range(len(X))]
        return [np.argpartition(dist_mat[i],len(k_t[i]))[:len(k_t[i])] for i in range(len(dist_mat))]

    k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0] for i in range(len(d))]
    k_embedded = get_k_embedded(X,k_theory)

    sum = 0
    for i in range(len(X)):
        count_intersect = 0
        for j in range(len(k_theory[i])):
            if k_theory[i][j] in k_embedded[i]:
                count_intersect += 1
        sum += count_intersect/(len(k_theory[i])+len(k_embedded[i])-count_intersect)

    return sum/len(X)

def get_k_nearest(d_row,k):
    return np.argpartition(d_row,k)[:k]


def k_nearest_embedded(X,k_theory):
    sum = 0
    dist_mat = np.zeros([len(X),len(X)])
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                dist_mat[i][j] = euclid_dist(X[i],X[j])
            else:
                dist_mat[i][j] = 100000
    k_embedded = [np.zeros(k_theory[i].shape) for i in range(len(k_theory))]

    for i in range(len(dist_mat)):
        k = len(k_theory[i])
        k_embedded[i] = np.argpartition(dist_mat[i],k)[:k]

    for i in range(len(X)):
        count_intersect = 0
        count_union = 0
        for j in range(len(k_theory[i])):
            if k_theory[i][j] in k_embedded[i]:
                count_intersect += 1
        sum += count_intersect/(len(k_theory[i])+len(k_embedded[i])-count_intersect)
    return sum/len(X)
