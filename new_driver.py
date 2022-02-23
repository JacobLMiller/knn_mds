from SGD_MDS2 import SGD_MDS2
from SGD_MDS import SGD_MDS

import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import matplotlib.pyplot as plt
import numpy as np
import graph_tool.all as gt

from metrics import get_neighborhood, get_norm_stress
from sklearn.metrics import pairwise_distances

graph = "rajat11"

G = gt.load_graph("graphs/{}.dot".format(graph))




def get_w(k=5,a=5):
    A = gt.adjacency(G).toarray()
    mp = np.linalg.matrix_power
    A = sum([mp(A,i) for i in range(1,a+1)])
    #A = np.linalg.matrix_power(A,a)

    A += np.random.normal(scale=0.01,size=A.shape)
    #A = 1-d_norm

    #k = 10
    k_nearest = [np.argpartition(A[i],-(k+1))[-(k+1):] for i in range(len(A))]

    n = G.num_vertices()
    N = 0
    w = np.asarray([[ 0 if i != j else 0 for i in range(len(A))] for j in range(len(A))])
    for i in range(len(A)):
        for j in k_nearest[i]:
            if i != j:
                w[i][j] = 1
                #w[j][i] = 1


    return w

d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

# Y = SGD_MDS(d)
# Xs = Y.solve(20,debug=True)
# X = layout_io.normalize_layout(Xs)
#
# print("Full SGD: Stress: {}, NP: {}".format(get_norm_stress(X,d_norm),get_neighborhood(X,d)))
# pos = G.new_vp('vector<float>')
# pos.set_2d_array(X.T)
# #
# gt.graph_draw(G,pos=pos)
#
#
# Y = SGD_MDS2(d,weighted=True,w=get_w(k=15,a=5))
# Xs = Y.solve(20,debug=True)
# X = layout_io.normalize_layout(Xs[-1])
#
# print("Local SGD: Stress: {}, NP: {}".format(get_norm_stress(X,d_norm),get_neighborhood(X,d)))
# pos = G.new_vp('vector<float>')
# pos.set_2d_array(X.T)
# #
# gt.graph_draw(G,pos=pos)
def opt_a(k=8):
    best_k = {}
    stress = {}
    for a in range(1,15):
        print(a)
        Y = SGD_MDS2(d,weighted=True,w=get_w(k=k,a=a))
        Xs = Y.solve(20,debug=True)
        X = layout_io.normalize_layout(Xs[-1])
        best_k[a] = get_neighborhood(X,d)
        stress[a] = get_norm_stress(X,d_norm)

    min_key = min(best_k, key = lambda k: best_k[k])
    print("The best k value I found was {} with an NP of {}".format(min_key,best_k[min_key]))
    keys = list(best_k.keys())
    print(keys.sort())
    x = [best_k[i] for i in keys]

    plt.plot( keys, x, label="NP")
    plt.plot( keys, [stress[i] for i in keys], label="stress")
    plt.legend()
    plt.show()

def opt_k(a=5):
    best_k = {}
    stress = {}
    for k in range(2,G.num_vertices(),8):
        print(k)
        Y = SGD_MDS2(d,weighted=True,w=get_w(k=k,a=a))
        Xs = Y.solve(20,debug=True)
        X = layout_io.normalize_layout(Xs[-1])
        best_k[a] = get_neighborhood(X,d)
        stress[a] = get_norm_stress(X,d_norm)

    min_key = min(best_k, key = lambda k: best_k[k])
    print("The best k value I found was {} with an NP of {}".format(min_key,best_k[min_key]))
    keys = list(best_k.keys())
    print(keys.sort())
    x = [best_k[i] for i in keys]

    plt.plot( keys, x, label="NP")
    plt.plot( keys, [stress[i] for i in keys], label="stress")
    plt.legend()
    plt.show()

opt_k(a=3)
