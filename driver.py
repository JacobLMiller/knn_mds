from SGD_MDS2 import SGD_MDS2
from MDS_classic import MDS
import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import numpy as np
import graph_tool.all as gt
import scipy.io

from tsnet_repeat import get_neighborhood,get_stress
from sklearn.metrics import pairwise_distances

def chen_neighborhood(D,X,k):
    embedded_dist = pairwise_distances(X)
    k_embed = []
    for i in range(len(embedded_dist)):
        k_embed.append(np.argsort(embedded_dist[i])[1:k+1])

    k_theory = []
    for i in range(len(D)):
        k_theory.append(np.argsort(D[i])[1:k+1])


    N_k = 0
    dKs = [k_theory[i][-1] for i in range(len(D))]
    for i in range(len(D)):
        dK = d[i][dKs[i]]
        inter = 0

        for j in range(len(k_embed[i])):
            if i != j:
                if embedded_dist[i][j] <= dK:
                    inter += 1
        N_k += inter
        #N_k += np.intersect1d(k_theory[i],k_embed[i]).size


    return N_k / (len(D)*k)

def avg_lcl_err(X,D):
    max_theory = np.max(D)

    embed = pairwise_distances(X)
    max_embed = np.max(embed)

    n = len(D)

    lcl_err = lambda i: sum([abs( (D[i][j] / max_theory)  - (embed[i][j] / max_embed) ) for j in range(n) if i != j ])/(n-1)
    return np.array([lcl_err(i) for i in range(n)])
    # for i in range(n):
    #     err = 0
    #     for j in range(n):
    #         if i != j:
    #             err += abs( (D[i][j] / max_theory)  - (embed[i][j] / max_embed) )



def my_random_graph(n,b,edge_probs):
    G = gt.Graph(directed=False)
    G.add_vertex(n)

    bm = G.new_vp('int')
    G.vertex_properties['bm'] = bm

    for v in G.vertices():
        bm[v] = random.randint(0,b-1)

    for i in range(G.num_vertices()):
        for j in range(i):
            if random.random() < edge_probs(bm[i],bm[j]):
                G.add_edge(i,j)
    return G,bm

def prob(a, b):

   if a == b:

       return 0.3

   else:

       return 0.01
import random
# G, bm = gt.random_graph(400, lambda: np.random.poisson(10), directed=False,
#
#                         model="blockmodel",
#
#                         block_membership=lambda: random.randint(0,4),
#
#                         edge_probs=prob)

G,bm = my_random_graph(50,2,prob)
#G.save('graphs/dummyblock.dot')

#G = gt.generate_sbm(list(bm), probs, out_degs=None, directed=False, micro_ers=False, micro_degs=False)


#G = gt.load_graph("graphs/fpga.dot")
#G = gt.lattice([10,10])
G = gt.load_graph('graphs/block2.dot')
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
    w = np.asarray([[ 0 if i != j else 0 for i in range(len(A))] for j in range(len(A))])
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
t = 0.7
T = np.linspace(0,1,10)
def power(n,count):
    for i in range(count):
        yield pow(n,i)

K = [x for x in power(2,12)]

A = gt.adjacency(G).toarray()
A = np.linalg.matrix_power(A,3)

# for k in K:
#     k = k if k < G.num_vertices() else G.num_vertices()
#     w = get_w(k)
#     for t in T:

w = get_w(k=4)
#w = gt.adjacency(G).toarray()

Y = SGD_MDS2(d,weighted=True,w=w)
Xs = Y.solve(num_iter=15,t=t,debug=True)
for layout in Xs:
    print(get_neighborhood(layout,d))

#print(get_neighborhood(Xs[-1],d,1))

X = layout_io.normalize_layout(Xs[-1])

d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True,verbose=False)


pos = G.new_vp('vector<float>')
pos.set_2d_array(X.T)

gt.graph_draw(G,pos=pos,vertex_fill_color=bm)


import matplotlib.pyplot as plt

percision = [chen_neighborhood(d_norm,X,k) for k in range(1,51)]
print(percision)

x =[i for i in range(1,51)]
plt.plot(x,percision)
plt.show()


# count = 0
# for layout in Xs:
#     X = layout_io.normalize_layout(layout)
#     #print(get_neighborhood(X,d))
#     #print(continuity(X,d))
#
#     pos = G.new_vp('vector<float>')
#     pos.set_2d_array(X.T)
#
#     gt.graph_draw(G,pos=pos,vertex_fill_color=bm,output='drawings/plot/test' + str(count) + '.png')
#     count += 1
#     #gt.graph_draw(G,pos=pos)
