from SGD_MDS2 import SGD
from SGD_MDS import SGD_MDS

import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import matplotlib.pyplot as plt
import numpy as np
import graph_tool.all as gt

from metrics import get_neighborhood, get_norm_stress, get_stress,get_cost
from sklearn.metrics import pairwise_distances

from SGD_MDS_debug import SGD_d


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
    print(k_nearest[0])

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
    Y = SGD_d(d,weighted=True,w=w)
    Xs = Y.solve(5000,debug=debug,radius=radius,t=t)
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
        gt.graph_draw(u,pos=pos,output=output)
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

def k_curve(graph,radius=False, n=5,folder='graphs/'):
    print(graph)

    G = gt.load_graph("{}{}.dot".format(folder,graph))
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

    diam = np.max(d)
    CC,_ = gt.global_clustering(G)
    a = 3

    K = np.linspace(10,100,10) if not radius else np.array([1,2,3,4,5,6,7])
    stress,NP = np.zeros(K.shape), np.zeros(K.shape)
    for _ in range(n):
        for i in range(len(K)):

            k = int(K[i])
            X,w = layout(G,d,d_norm,debug=False,k=k,a=a,radius=radius)
            draw(G,X,output='drawings/{}_k{}.png'.format(graph,k))

            stress[i] += get_stress(X,d_norm)
            NP[i] += (get_neighborhood(X,d))
    stress = stress/n
    NP = NP/n


    plt.plot(K.astype(int), stress, label="Stress")
    plt.plot(K.astype(int), NP, label="NP")
    plt.suptitle(graph)
    plt.xlabel("k")
    plt.legend()
    plt.savefig('figures/kcurve_{}.png'.format(graph))
    plt.clf()
    print()

def a_curve(graph,radius=False, n=5):
    print(graph)

    G = gt.load_graph("graphs/{}.dot".format(graph))
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

    diam = np.max(d)
    CC,_ = gt.global_clustering(G)

    A = np.array([1,2,3,4,5,6,7,8])
    stress,NP = np.zeros(A.shape), np.zeros(A.shape)
    for _ in range(n):
        for i in range(len(A)):
            a = A[i]
            X,w = layout(G,d,d_norm,debug=True,k=22,a=a,radius=radius)
            draw(G,X[-1],output='drawings/{}_a{}.png'.format(graph,a))

            stress[i] += get_stress(X[-1],d_norm)
            NP[i] += (get_neighborhood(X[-1],d))
    stress = stress/n
    NP = NP/n


    plt.plot(A.astype(int), stress, label="Stress")
    plt.plot(A.astype(int), NP, label="NP")
    plt.suptitle(graph)
    plt.xlabel("a")
    plt.legend()
    plt.savefig('figures/acurve_{}.png'.format(graph))
    plt.clf()
    print()


def t_curve(graph,radius=False, n=5):
    print(graph)

    G = gt.load_graph("random_runs/{}.dot".format(graph))
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

    diam = np.max(d)
    CC,_ = gt.global_clustering(G)

    A = np.linspace(0,1,8)
    stress,NP = np.zeros(A.shape), np.zeros(A.shape)
    for _ in range(n):
        for i in range(len(A)):
            t = A[i]
            X,w = layout(G,d,d_norm,debug=True,k=22,a=3,t=t,radius=radius)
            draw(G,X[-1],output='drawings/{}_t{}.png'.format(graph,t))

            stress[i] += get_stress(X[-1],d_norm)
            NP[i] += (get_neighborhood(X[-1],d))
    stress = stress/n
    NP = NP/n


    plt.plot(A.astype(int), stress, label="Stress")
    plt.plot(A.astype(int), NP, label="NP")
    plt.suptitle(graph)
    plt.xlabel("t")
    plt.legend()
    plt.savefig('figures/tcurve_{}.png'.format(graph))
    plt.clf()
    print()

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

    K = np.linspace( 5,G.num_vertices()-1, 8)

    w = get_w(G,k=k,a=a)

    cost = []

    for t in range(30,201,10):
        Y = SGD(d,weighted=True, w = w)
        X = Y.solve(t,debug=False,t=0.1)

        print('NP: {}'.format(get_neighborhood(X,d)))
        print('stress: {}'.format(get_stress(X,d)))
        draw(G,X,output='test{}.png'.format(t))
        cost.append(get_cost(X,d,w,0.1))

    plt.plot(list(range(30,201,10)),cost,label='Cost value')
    plt.show()

def drive(graph,hist=False,output=None,k=10):
    G = gt.load_graph("{}.dot".format(graph))
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

    a = 5

    w = get_w(G,k=k,a=a)


    Y = SGD(d,weighted=True, w = w)
    X = Y.solve(60,debug=hist,t=0.1)
    X = layout_io.normalize_layout(X)
    print('NP: {}'.format(get_neighborhood(X,d,1)))
    print('stress: {}'.format(get_stress(X,d)))
    #X = [x for x in X]
    if hist:
        draw_hist(G,X,d,w,Y)
    else:
        draw(G,X,output=output)

if __name__ == '__main__':
    for k in [10,22,48,74,100]:
        drive('graphs/jazz',k=k,output='jazz_k{}.png'.format(k))

    drive('graphs/btree9',hist=False)
    #iterate('random_runs/block_model_200')
    #drive('graphs/dwt_419',hist=False)
    # import cProfile
    # cProfile.run('drive(\'graphs/10square\',hist=False,radius=False)')

    #k_curve('airlines',folder='graphs/',radius=False,n=5)
    #a_curve('test_mnist',radius=False,n=3)
    #t_curve('custom_cluster_300',radius=False,n=3)
