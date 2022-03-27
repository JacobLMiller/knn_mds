from simple_driver import get_w
from metrics import get_cost, get_neighborhood, get_stress
from SGD_MDS2 import SGD

import numpy as np
import graph_tool.all as gt

import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import matplotlib.pyplot as plt

def iteration_exp(n,G,d,d_norm,a,graph):
    t_max = [i for i in range(15,121,15)]
    iteration = np.zeros(len(t_max))

    k = 21

    w = get_w(G,k=k,a=a)

    for i in range(n):
        for t in range(len(t_max)):
            Y = SGD(d,weighted=True, w = w)
            X = Y.solve(t_max[t],t=0.1)
            #X = layout_io.normalize_layout(X)

            iteration[t] += get_cost(X,d,w,0.1)
            #iteration[t] += get_stress(X,d_norm)

    iteration /= n

    plt.plot(t_max,iteration)
    plt.savefig('figures/max_iter_{}'.format(graph))
    plt.clf()


    return iteration


def matrix_exp(n,G,d,d_norm):
    d_power = [d for d in range(1,13)]
    powers = np.zeros(len(d_power))

    k = 21


def experiment(n=5):
    import os
    import pickle
    import copy

    path = 'example_graphs/'
    graph_paths = os.listdir(path)
    #graph_paths = ['block_400.dot']

    graph_paths = list( map(lambda s: s.split('.')[0], graph_paths) )
    #graph_paths = ['custom_cluster_100']
    print(graph_paths)

    adjacency_len = len( np.linspace(5,100,8) )

    zeros = lambda s: np.zeros(s)
    alg_dict = {'NP': zeros(adjacency_len), 'stress': zeros(adjacency_len), 'cost': zeros(adjacency_len)}
    exp = {'iterations': copy.deepcopy(alg_dict),
           'matrix_power': copy.deepcopy(alg_dict),
           'alpha': copy.deepcopy(alg_dict)}


    graph_dict = {key: copy.deepcopy(exp) for key in graph_paths}

    for graph in graph_paths:
        print(graph)
        G = gt.load_graph(path+graph + '.dot')
        d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
        d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

        CC = G.num_edges() // G.num_vertices()
        a = 3 if CC < 4 else 4 if CC < 8 else 5

        print("Graph: " + graph)
        print("-----------------------------------------------------------")


        graph_dict['iterations'] = iteration_exp(n,G,d,d_norm,a,graph)
        #graph_dict['matrix_power'] = matrix_exp(n,G,d,d_norm)


if __name__ == '__main__':
    experiment(n=5)
