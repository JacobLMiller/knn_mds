import numpy as np
import graph_tool.all as gt
import modules.layout_io as layout_io
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne

#from SGD_MDS2 import SGD_MDS2
from SGD_MDS_debug import SGD_d
from SGD_MDS import SGD_MDS

from sklearn.metrics import pairwise_distances

from metrics import get_stress,get_neighborhood

import matplotlib.pyplot as plt

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
                w[j][i] = 1

    return w

def draw(G,X,output=None):
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    #
    if output:
        gt.graph_draw(G,pos=pos,output=output)
    else:
        gt.graph_draw(G,pos=pos)

def calc_adj(graph,G,d,d_norm):
    NP,stress = [],[]
    K = np.linspace( 5,100, 12)
    a = 3 if graph[0] == 'p' else 5
    for k in K:
        k = int(k)
        w = get_w(G,k=k,a=a)
        Y = SGD_d(d,w=w,weighted=True)
        sol = [x for x in Y.solve(1500,debug=True,radius=False)]
        Xs = sol[-1]
        X = layout_io.normalize_layout(Xs)
        stress.append(get_stress(X,d_norm))
        NP.append(get_neighborhood(X,d))

        draw(G,X,output='drawings/random_graphs/adjacency_power/{}_k{}.png'.format(graph,k))


    plt.plot(K, stress, label="Stress")
    plt.plot(K, NP, label="NP")
    plt.suptitle(graph)
    plt.xlabel("k")
    plt.legend()
    plt.savefig('figures/adjacency_kcurve_{}.png'.format(graph))
    plt.clf()

    return np.array(NP),np.array(stress)

def calc_radius(graph,G,d,d_norm):
    NP,stress = [],[]
    K = list(range(1,8+1))
    for k in K:
        w = get_w(G,k=k,a=5)
        Y = SGD_d(d,weighted=True,w=w)
        sol = [x for x in Y.solve(1500,debug=True,radius=True)]
        Xs = sol[-1]
        X = layout_io.normalize_layout(Xs)
        stress.append(get_stress(X,d_norm))
        NP.append(get_neighborhood(X,d))

        draw(G,X,output='drawings/random_graphs/radius/{}_k{}.png'.format(graph,k))


    plt.plot(K, stress, label="Stress")
    plt.plot(K, NP, label="NP")
    plt.suptitle(graph)
    plt.xlabel("k")
    plt.legend()
    plt.savefig('figures/radius_kcurve_{}.png'.format(graph))
    plt.clf()

    return np.array(NP),np.array(stress)

def calc_linear(graph,G,d,d_norm):
    NP,stress = [],[]
    A = np.linspace(0,1,12)
    for a in A:

        n = 1500
        momentum = 0.5
        tolerance = 1e-7
        window_size = 40
        # Cost function parameters
        r_eps = 0.05
        # Phase 2 cost function parameters
        lambdas_2 = [1, 1.2, 0]
        # Phase 3 cost function parameters
        lambdas_3 = [1, 0.01, 0.6]

        Y,hist = thesne.tsnet(
            d, output_dims=2, random_state=1, perplexity=40, n_epochs=n,
            Y=None,
            initial_lr=50, final_lr=50, lr_switch=n // 2,
            initial_momentum=momentum, final_momentum=momentum, momentum_switch=n // 2,
            initial_l_kl=lambdas_2[0], final_l_kl=lambdas_3[0], l_kl_switch=n // 2,
            initial_l_c=lambdas_2[1], final_l_c=lambdas_3[1], l_c_switch=n // 2,
            initial_l_r=lambdas_2[2], final_l_r=lambdas_3[2], l_r_switch=n // 2,
            r_eps=r_eps, autostop=tolerance, window_size=window_size,
            verbose=True, a=a
        )

        X = layout_io.normalize_layout(Y)
        stress.append(get_stress(X,d_norm))
        NP.append(get_neighborhood(X,d))

        draw(G,X,output='drawings/random_graphs/linear_comb/{}_a{}.png'.format(graph,a))


    plt.plot(A, stress, label="Stress")
    plt.plot(A, NP, label="NP")
    plt.suptitle(graph)
    plt.xlabel("a")
    plt.legend()
    plt.savefig('figures/linear_kcurve_{}.png'.format(graph))
    plt.clf()

    return np.array(NP),np.array(stress)



def main(n=5):
    import os
    import pickle
    import copy

    path = 'random_runs/'
    graph_paths = os.listdir(path)

    graph_paths = list( map(lambda s: s.split('.')[0], graph_paths) )
    print(graph_paths)

    adjacency_len = len( np.linspace(5,100,8) )
    radius_len = len( list(range(1,8+1) ))
    linear_len = len( np.linspace(0,1,8) )

    zeros = lambda s: np.zeros(s)
    alg_dict = {'adjacency_power': {'NP': zeros(adjacency_len), 'stress': zeros(adjacency_len)},
                'radius': {'NP': zeros(radius_len), 'stress': zeros(radius_len)},
                'linear': {'NP': zeros(linear_len), 'stress': zeros(linear_len)}
                }

    graph_dict = {key: copy.deepcopy(alg_dict) for key in graph_paths}

    for graph in graph_paths:
        print(graph)
        G = gt.load_graph(path+graph + '.dot')
        d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
        d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)
        #if G.num_vertices() > 999: continue

        CC,_ = gt.global_clustering(G)
        a = 2 if CC < 0.1 else 3 if CC < 0.4 else 4 if CC < 0.6 else 5

        print("Graph: " + graph)
        print("-----------------------------------------------------------")

        for i in range(n):
            print("Iteration number ", i)

            NP,stress = calc_adj(graph,G,d,d_norm)
            graph_dict[graph]['adjacency_power']['NP'] += NP
            graph_dict[graph]['adjacency_power']['stress'] += stress

            NP,stress = calc_linear(graph,G,d,d_norm)
            graph_dict[graph]['linear']['NP'] += NP
            graph_dict[graph]['linear']['stress'] += stress

        for alg in alg_dict.keys():
            for metric in ['NP','stress']:
                graph_dict[graph][alg][metric] /= n

        with open('data/random_graphs1.pkl','wb') as myfile:
            pickle.dump(graph_dict,myfile)
        myfile.close()


if __name__ == "__main__":
    main(n=30)
