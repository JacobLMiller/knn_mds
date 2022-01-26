from SGD_MDS2 import SGD_MDS2
from MDS_classic import MDS
import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import numpy as np
import graph_tool.all as gt
import scipy.io

from tsnet_repeat import get_neighborhood
from sklearn.metrics import pairwise_distances

def continuity(X,d,rg = 2):
    """
    How well do the local neighborhoods represent the theoretical neighborhoods?
    Closer to 1 is better.
    Measure of percision: ratio of true positives to true positives+false positives
    """
    norm = np.linalg.norm
    def get_k_embedded(X,k_t):
        dist_mat = pairwise_distances(X)
        return [np.argpartition(dist_mat[i],len(k_t[i]))[:len(k_t[i])] for i in range(len(dist_mat))]

    k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0] for i in range(len(d))]

    k_embedded = get_k_embedded(X,k_theory)

    sum = 0
    n = len(X)
    for i in range(n):
        k = len(k_theory[i])
        rank = 0
        print(len(k_theory[i]))
        for j in range(len(k_theory[i])):
            if k_theory[i][j] not in k_embedded[i]:
                rank += d[i][j] - len(k_theory[i])
                print(k_theory[i][j])
        print()
        bottom = n*k*(2*n - 3*k -1)
        sum += rank*(2/bottom)

    return 1-sum

def prob(a, b):

   if a == b:

       return 0.999999

   else:

       return 0.000001
import random
G, bm = gt.random_graph(400, lambda: np.random.poisson(lam=10), directed=False,

                        model="blockmodel",

                        block_membership=lambda: random.randint(0,4),

                        edge_probs=prob)


#G = gt.load_graph("graphs/fpga.dot")
# G = gt.lattice([10,40])
#G = gt.load_graph('graphs/block_2000.dot')
#G = gt.load_graph('graphs/btree8.dot')
d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)

def get_w(k=5):
    A = gt.adjacency(G).toarray()
    A = np.linalg.matrix_power(A,5)
    A += np.random.normal(scale=0.01,size=A.shape)

    #k = 10
    k_nearest = [np.argpartition(A[i],-k)[-k:] for i in range(len(A))]

    n = G.num_vertices()
    N = 0
    w = np.asarray([[ 1e-5 if i != j else 0 for i in range(len(A))] for j in range(len(A))])
    for i in range(len(A)):
        for j in k_nearest[i]:
            if i != j:
                w[i][j] = 1
                w[j][i] = 1
    return w

# for i in range(len(w)):
#     for j in range(i):
#         if w[i][j] == 1:
#             N += 1

# Nc = (n*(n-1))/2 - N
#t = (N/Nc)*np.median(d)*0.1
t = 0.3
T = np.linspace(0,1,10)
def power(n,count):
    for i in range(count):
        yield pow(n,i)

K = [x for x in power(2,12)]
print(K)
A = gt.adjacency(G).toarray()
A = np.linalg.matrix_power(A,3)

# for k in K:
#     k = k if k < G.num_vertices() else G.num_vertices()
#     w = get_w(k)
#     for t in T:

w = get_w(k=8)
#w = gt.adjacency(G).toarray()
t = 0.99999

Y = MDS(d,weighted=True,w=w)
Xs = Y.solve(num_iter=1000,t=t,debug=False)

X = layout_io.normalize_layout(Xs)
print(get_neighborhood(X,d))
#print(continuity(X,d))

pos = G.new_vp('vector<float>')
pos.set_2d_array(X.T)

#gt.graph_draw(G,pos=pos,output='drawings/dwt_k' + str(k) + 't' + str(round(t,2)) + '.png')
gt.graph_draw(G,pos=pos)
