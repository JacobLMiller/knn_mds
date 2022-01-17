import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from math import sqrt
import itertools
import scipy
from numba import jit

import math
import random

norm = lambda x: np.linalg.norm(x,ord=2)



class SGD_MDS2:
    def __init__(self,dissimilarities,k=5,weighted=False,w = np.array([]), init_pos=np.array([])):
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
            self.w = w
        else:
            self.w = np.array([[ 1 if i != j else 0 for i in range(self.n)]
                        for j in range(self.n)])

        w_min = 1/pow(self.d_max,2)
        self.w_max = 1/pow(self.d_min,2)
        self.eta_max = 1/w_min
        epsilon = 0.1
        self.eta_min = epsilon/self.w_max



    def calc_stress(self):
        stress = 0
        for i in range(self.n):
            for j in range(i):
                stress += self.w[i][j]*pow(geodesic(self.X[i],self.X[j])-self.d[i][j],2)
        return pow(stress,0.5)

    def calc_distortion(self):
        distortion = 0
        for i in range(self.n):
            for j in range(i):
                distortion += abs((norm(self.X[i]-self.X[j])-self.d[i][j]))/self.d[i][j]
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
        #return 1/pow(5+count,1)

        lamb = math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return self.eta_max*math.exp(lamb*count)


    def init_point(self):
        return [random.uniform(-1,1),random.uniform(-1,1)]


@jit(nopython=True)
def satisfy(v,u,di,step):

    if True:

        wc = step
        pq = v-u #Vector between points
        mag = norm(pq)

        wc = 1*step
        if wc > 1:
            wc = 1

        # if we < 0.9 and random.random()<0.1:
        #     r = (mag-10*self.d_max)/2
        #     wc = step
        # elif we > 0.9:

        r = (mag-di)/2
        r = wc*r
        m = (pq*r) /mag

        return v-m, u+m

    elif di != 1:
        pq = v-u
        mag = np.linalg.norm(pq)
        r = mag/2
        step = step if step <= 1 else 1
        m = (pq*r*step) /mag

        m *= 1
        return v-m, u+m

@jit(nopython=True)
def step_func(count):
    return 1/(5+count)

@jit(nopython=True)
def solve(X,d,indices,num_iter=15,epsilon=1e-3,debug=False):

    step = 1
    shuffle = random.shuffle

    for count in range(num_iter):
        for i,j in indices:
            X[i],X[j] = satisfy(X[i],X[j],d[i][j],step)

        step = step_func(count)
        shuffle(indices)

    return X

def normalize(v):
    mag = pow(v[0]*v[0]+v[1]*v[1],0.5)
    return v/mag

def geodesic(xi,xj):
    return euclid_dist(xi,xj)

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product


def bfs(G,start):
    queue = [start]
    discovered = [start]
    distance = {start: 0}

    while len(queue) > 0:
        v = queue.pop()

        for w in G.neighbors(v):
            if w not in discovered:
                discovered.append(w)
                distance[w] =  distance[v] + 1
                queue.insert(0,w)

    myList = []
    for x in G.nodes:
        if x in distance:
            myList.append(distance[x])
        else:
            myList.append(len(G.nodes)+1)

    return myList

def all_pairs_shortest_path(G):
    d = [ [ -1 for i in range(len(G.nodes)) ] for j in range(len(G.nodes)) ]

    count = 0
    for node in G.nodes:
        d[count] = bfs(G,node)
        count += 1
    return d

def scale_matrix(d,new_max):
    d_new = np.zeros(d.shape)

    t_min,r_min,r_max,t_max = 0,0,np.max(d),new_max

    for i in range(d.shape[0]):
        for j in range(i):
            m = ((d[i][j]-r_min)/(r_max-r_min))*(t_max-t_min)+t_min
            d_new[i][j] = m
            d_new[j][i] = m
    return d_new


def euclid_dist(x1,x2):
    x = x2[0]-x1[0]
    y = x2[1]-x1[1]
    return pow(x*x+y*y,0.5)

def save_euclidean(X,number):
    pos = {}
    count = 0
    for i in G.nodes():
        x,y = X[count]
        pos[i] = [x,y]
        count += 1
    nx.draw(G,pos=pos,with_labels=True)
    plt.savefig('test'+str(number)+'.png')
    plt.clf()

def output_euclidean(G,X):
    pos = {}
    count = 0
    for x in G.nodes():
        pos[x] = X[count]
        count += 1
    nx.draw(G,pos=pos)
    plt.show()
    plt.clf()

    count = 0
    for i in G.nodes():
        x,y = X[count]
        G.nodes[i]['pos'] = str(100*x) + "," + str(100*y)

        count += 1
    nx.drawing.nx_agraph.write_dot(G, "output.dot")

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
