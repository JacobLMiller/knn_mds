from SGD_MDS2 import SGD_MDS2
from MDS_classic import MDS
import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import numpy as np
import graph_tool.all as gt
import scipy.io

from tsnet_repeat import get_neighborhood


#G = gt.load_graph("graphs/dwt_419.dot")
#G = gt.lattice([10,10])
G = gt.load_graph('graphs/price_1000.dot')
#G = gt.load_graph('graphs/btree8.dot')
d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)

A = gt.adjacency(G).toarray()
A = np.linalg.matrix_power(A,5)
A += np.random.normal(scale=0.01,size=A.shape)

k = 3
k_nearest = [np.argpartition(A[i],-k)[-k:] for i in range(len(A))]

n = G.num_vertices()
N = 0
w = np.asarray([[ 1e-5 if i != j else 0 for i in range(len(A))] for j in range(len(A))])
for i in range(len(A)):
    for j in k_nearest[i]:
        if i != j:
            w[i][j] = 1
            w[j][i] = 1

for i in range(len(w)):
    for j in range(i):
        if w[i][j] == 1:
            N += 1

Nc = (n*(n-1))/2 - N
#t = (N/Nc)*np.median(d)*0.1
t = 1

Y = SGD_MDS2(d*10,weighted=False,w=w)
Xs = Y.solve(num_iter=30,t=t,debug=False)

X = layout_io.normalize_layout(Xs)
print(get_neighborhood(X,d))

pos = G.new_vp('vector<float>')
pos.set_2d_array(X.T)

gt.graph_draw(G,pos=pos)
