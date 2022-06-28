from SGD_MDS2 import SGD

import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import matplotlib.pyplot as plt
import numpy as np
import graph_tool.all as gt

from metrics import get_neighborhood, get_stress,get_cost
from sklearn.metrics import pairwise_distances


def get_w(G,k=5,a=5,eps=0):
    A = gt.adjacency(G).toarray()
    mp = np.linalg.matrix_power
    A = sum([mp(A,i) for i in range(1,a+1)])
    #A = np.linalg.matrix_power(A,a)

    #A += np.random.normal(scale=0.01,size=A.shape)
    #A = 1-d_norm

    #k = 10
    B = np.argsort(A,axis=1)
    k_nearest = [[None for _ in range(k)] for _ in range(len(A))]
    for i in range(len(A)):
        A_star = B[i][::-1]
        for j in range(k):
            if A[i][A_star[j]] == 0: break
            k_nearest[i][j] = A_star[j]
        if j < k:
            pass

    #k_nearest = [np.argpartition(A[i],-(k+1))[-(k+1):] for i in range(len(A))]
    #print(A[0][k_nearest[0]])

    n = G.num_vertices()
    N = 0
    w = np.asarray([[ eps if i != j else 0 for i in range(len(A))] for j in range(len(A))])
    for i in range(len(A)):
        for j in k_nearest[i]:
            if i != j and j:
                w[i][j] = 1
                w[j][i] = 1

    return w

def layout(G,d,d_norm,debug=True, k=7, a=5,t=0.6,radius=False):
    k = k if k < G.num_vertices() else G.num_vertices()
    w = get_w(G,k=k,a=a)
    Y = SGD(d,weighted=True,w=w)
    Xs = Y.solve(60,debug=debug,radius=radius,t=t)
    X = layout_io.normalize_layout(Xs)

    return X,w


def draw(G,X,output=None):
    from graph_tool import GraphView

    b = 50

    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    #
    u = GraphView(G, efilt=lambda e: True)
    deg = G.degree_property_map("total")
    deg.a = 4 * (np.sqrt(deg.a) * 0.5 + 0.4)


    if output:
        gt.graph_draw(u,pos=pos,output=output)#,vertex_fill_color=color)
    else:
        gt.graph_draw(G,pos=pos)

def layout_directory():
    import os
    graph_paths = os.listdir('new_tests/')
    for graph in graph_paths:
        k_curve(graph.split('.')[0])

def draw_hist(G,Xs,d,w,Y):
    NP = []
    cost = []


    for count in range(len(Xs)-1):
        if count % 1 == 0: #or count < 100:
            draw( G,Xs[count],output="drawings/update/test_{}.png".format(count) )
            NP.append(get_neighborhood(Xs[count],d))
            cost.append( get_cost( Xs[count], d, w, 0.6 ) )


    print("NP: ", NP[-1])
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(NP)),NP)
    plt.show()
    plt.clf()
    plt.suptitle("Cost function")
    plt.plot(np.arange(len(cost)),cost)
    plt.show()


def iterate(graph):
    G = gt.load_graph("{}.dot".format(graph))
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

    a = 5
    k = 10

    K = list(range(10,101,10))



    cost,NP,stress = [],[],[]

    for k in K:
        w = get_w(G,k=k,a=a)
        Y = SGD(d,weighted=True, w = w)
        X = Y.solve(60,debug=False,t=0.1)

        NP.append(get_neighborhood(X,d))
        stress.append(get_stress(X,d))
        draw(G,X,output='test{}.png'.format(k))
        cost.append(get_cost(X,d,w,0.1))

    plt.plot(NP,stress,marker='o')
    for xy in zip(NP,stress):                                       # <--
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--

    plt.show()
    plt.clf()

    plt.plot(K,stress)
    plt.plot(K,NP)
    plt.show()

def drive(graph,L2G=False,hist=False,output=None,k=10):
    G = gt.load_graph("{}.dot".format(graph))

    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    #d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)


    w = get_w(G,k=k)


    Y = SGD(d,weighted=L2G, w = w)
    X = Y.solve(60,debug=hist,t=0.1)
    X = layout_io.normalize_layout(X)

    print('NP: {}'.format(get_neighborhood(X,d,2)))
    print('stress: {}'.format(get_stress(X,d)))

    if hist:
        draw_hist(G,X,d,w,Y)
    else:
        draw(G,X,output=output)

if __name__ == '__main__':
    drive('graphs/10square',L2G=False)
