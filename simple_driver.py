from SGD_MDS2 import SGD_MDS2
from SGD_MDS import SGD_MDS

import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import matplotlib.pyplot as plt
import numpy as np
import graph_tool.all as gt

from metrics import get_neighborhood, get_norm_stress, get_stress
from sklearn.metrics import pairwise_distances

def layout(G,d,d_norm,debug=False,k=8,a=5):
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


    Y = SGD_MDS2(d,weighted=True,w=get_w(k=k,a=a))
    Xs = Y.solve(20,debug=debug)

    if debug:
        Xs = [layout_io.normalize_layout(X) for X in Xs]
        print("Local SGD: Stress: {}, NP: {}".format(get_norm_stress(Xs[-1],d_norm),get_neighborhood(Xs[-1],d)))
        return Xs
    else:
        return Xs


def draw(G,X,output=None):
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    #
    if output:
        gt.graph_draw(G,pos=pos,output=output)
    else:
        gt.graph_draw(G,pos=pos)

def stress_curve():
    graph = 'dwt_419'

    G = gt.load_graph("graphs/{}.dot".format(graph))
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)


    X = layout(G,d,d_norm,debug=True)
    draw(G,X[-1])

    stress, NP = [get_norm_stress(x,d_norm) for x in X], [get_neighborhood(x,d) for x in X]
    plt.plot(np.arange(len(stress)), stress, label="Stress")
    plt.plot(np.arange(len(stress)), NP, label="NP")
    plt.suptitle("Block graph")
    plt.legend()
    plt.show()

def k_curve(graph):
    print(graph)

    G = gt.load_graph("graphs/{}.dot".format(graph))
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)
    if G.num_vertices() > 2001:
        return

    diam = np.max(d)
    CC,_ = gt.global_clustering(G)
    a = 2 if diam > 22 else 3 if diam > 12 else 4 if diam > 8 else 5

    K = np.linspace(2,100,12)
    stress,NP = [], []
    for k in K:

        k = int(k)
        X = layout(G,d,d_norm,debug=True,k=k,a=a)
        draw(G,X[-1],output='drawings/{}_k{}.png'.format(graph,k))

        stress.append(get_norm_stress(X[-1],d_norm))
        NP.append(get_neighborhood(X[-1],d))


    plt.plot(K.astype(int), stress, label="Stress")
    plt.plot(K.astype(int), NP, label="NP")
    plt.suptitle(graph)
    plt.xlabel("k")
    plt.legend()
    plt.savefig('figures/kcurve_{}.png'.format(graph))
    plt.clf()
    print()

def layout_directory():
    import os
    graph_paths = os.listdir('tsnet-graphs/')
    for graph in graph_paths:
        k_curve(graph.split('.')[0])

def draw_hist(G,Xs):
    for count in range(len(Xs)):
        draw(G,Xs[count],output="drawings/update/test_{}.png".format(count))

def drive(graph,hist=False):
    G = gt.load_graph("graphs/{}.dot".format(graph))
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

    Xs = layout(G,d,d_norm,debug=True, k=2, a=5)
    if hist:
        draw_hist(G,Xs)
    else:
        draw(G,Xs[-1])


drive('10square',hist=True)
