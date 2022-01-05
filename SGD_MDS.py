import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from math import sqrt
import itertools



import math
import random

norm = lambda x: np.linalg.norm(x,ord=2)


class SGD_MDS:
    def __init__(self,dissimilarities,k=5,weighted=False,w = np.array([]), init_pos=np.array([])):
        self.d = dissimilarities
        self.d_max = np.max(dissimilarities)
        print(self.d_max)
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
        sym = True
        for i in range(len(self.w)):
            for j in range(i):
                if self.w[i][j] != self.w[j][i]:
                    print(self.w[i][j])
                    print(self.w[j][i])
                    print()
                    sym = False
        print('w is symmetric? ', sym)
        if not sym:
            print(self.w)
        w_min = 1/pow(self.d_max,2)
        self.w_max = 1/pow(self.d_min,2)
        self.eta_max = 1/w_min
        epsilon = 0.1
        self.eta_min = epsilon/self.w_max


    def solve(self,num_iter=15,epsilon=1e-3,debug=False):
        current_error,delta_e,step,count = 1000,1,self.eta_max,0
        #indices = [i for i in range(len(self.d))]
        indices = list(itertools.combinations(range(self.n), 2))
        random.shuffle(indices)
        #random.shuffle(indices)

        max_move = 0
        self.hist = [self.X]
        self.stress_hist = []
        self.neighbor_hist = []
        #print('neighborhood: ', get_neighborhood(self.X,self.d))
        print('stress', self.calc_stress())
        print()
        if debug:
            #self.stress_hist.append(self.calc_stress())
            self.hist.append(self.X.copy())

        while count < num_iter:
            #print('Epoch: {0}.'.format(count), end='\r')

            for k in range(len(indices)):

                i = indices[k][0]
                j = indices[k][1]
                if i > j:
                    i,j = j,i
                # i = random.randint(0,self.n-1)
                # j = random.randint(0,self.n-1)
                # if i == j:
                #     j = j + 1 if j != self.n-1 else j-1

                wc = step


                pq = self.X[i] - self.X[j] #Vector between points

                #mag = geodesic(self.X[i],self.X[j])
                mag = norm(pq)
                r = (mag-self.d[i][j])/2 #min distance to satisfy constraint
                wc = self.w[i][j]*step

                if self.w[i][j] < 0.9 and random.random()<0.1:
                    r = (mag-10*self.d_max)/2
                    wc = step
                elif self.w[i][j] > 0.9:
                    r = (mag-(self.d[i][j]))/2

                #/pow(self.d[i][j],2)
                if wc > 1:
                    wc = 1
                r = wc*r

                m = (pq*r) /mag

                self.X[i] = self.X[i] - m
                self.X[j] = self.X[j] + m

                nmag = norm(self.X[i]-self.X[j])
                max_move = max(abs(nmag-mag),max_move)

                #save_euclidean(self.X,weight)
                #weight += 1

            if max_move < epsilon:
                break
            step = self.compute_step_size(count,num_iter)
            #step = 1

            count += 1
            random.shuffle(indices)
            if debug:
                #print(self.calc_distortion())
                self.hist.append(self.X.copy())
                neighbor = get_neighborhood(self.X,self.d)
                print('neighborhood: ',neighbor)
                stress = self.calc_stress()
                print('stress', stress)
                self.stress_hist.append(stress)
                self.neighbor_hist.append(neighbor)
                print()

        #print("Total epochs: {0}. Final Stress: {1}".format(count,self.calc_stress()))
        #print(get_neighborhood(self.X,self.d))
        return self.X

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
        return 1/pow(5+count,1)

        lamb = math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return self.eta_max*math.exp(lamb*count)


    def init_point(self):
        return [random.uniform(-1,1),random.uniform(-1,1)]


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
