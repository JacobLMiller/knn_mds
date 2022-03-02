from SGD_MDS2 import SGD_MDS2, calc_cost
from SGD_MDS import SGD_MDS

import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import matplotlib.pyplot as plt
import numpy as np
import graph_tool.all as gt

from metrics import get_neighborhood, get_norm_stress, get_stress
from sklearn.metrics import pairwise_distances

def get_w(G,k=5,a=5):
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

def layout(G,d,d_norm,debug=True, k=7, a=5):
    k = k if k < G.num_vertices() else G.num_vertices()
    w = get_w(G,k=k,a=a)
    Y = SGD_MDS2(d,weighted=True,w=w)
    Xs = Y.solve(100,debug=debug)

    if debug:
        #Xs = [layout_io.normalize_layout(X) for X in Xs]
        print("Local SGD: Stress: {}, NP: {}".format(get_norm_stress(Xs[-1],d_norm),get_neighborhood(Xs[-1],d)))
        return Xs,w
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

    diam = np.max(d)
    CC,_ = gt.global_clustering(G)
    a = 3

    K = np.linspace(5,G.num_vertices()-1,7)
    stress,NP = [], []
    for k in K:

        k = int(k)
        X,w = layout(G,d,d_norm,debug=True,k=k,a=a)
        draw(G,X[-1],output='drawings/{}_k{}.png'.format(graph,k))

        stress.append(get_stress(X[-1],d_norm))
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
    graph_paths = os.listdir('new_tests/')
    for graph in graph_paths:
        k_curve(graph.split('.')[0])

def draw_hist(G,Xs,d,w):
    NP = []
    cost = []
    for count in range(len(Xs)-1):
        if count % 100 == 0 or count < 100:
            draw( G,Xs[count],output="drawings/update/test_{}.png".format(count) )
            NP.append(get_neighborhood(Xs[count],d))
            cost.append( calc_cost( Xs[count], d, w, 0.6 ) )



    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(NP)),NP)
    plt.show()
    plt.clf()
    plt.suptitle("Cost function")
    plt.plot(np.arange(len(cost)),cost)
    plt.show()



def drive(graph,hist=False):
    G = gt.load_graph("graphs/{}.dot".format(graph))
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

    Xs,w = layout(G,d,d_norm,debug=True, k=2, a=1)
    if hist:
        draw_hist(G,Xs,d,w)
    else:
        draw(G,Xs[-2])

def drive_new(graph,hist=False):
    G = gt.load_graph("graphs/{}.dot".format(graph))
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

    w = get_w(G,k=2,a=3)

    from SGD_MDS_debug import SGD_d
    Y = SGD_d(d,weighted=True, w = w)
    X = Y.solve(1000)
    X = [x for x in X]
    if hist:
        draw_hist(G,X,d,w)
    else:
        draw(G,X)


drive('dwt_419',hist=True)
#k_curve('block2')
