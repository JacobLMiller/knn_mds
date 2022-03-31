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

    plt.plot(t_max,iteration,'g-',label='Cost function')
    plt.suptitle("Maximum iteration experiment on {}".format(graph))
    plt.xlabel("Max_iter")
    plt.ylabel("Cost value")
    plt.legend()
    plt.savefig('figures/max_iter_{}.eps'.format(graph))
    plt.clf()


    return iteration


def matrix_exp(n,G,d,d_norm,graph):
    d_power = [a for a in range(1,13)]
    powers = {'NP': np.zeros(len(d_power)), 'stress': np.zeros(len(d_power))}

    k = 21

    for i in range(n):
        for a in range(len(d_power)):
            w = get_w(G,k=k,a=d_power[a])
            Y = SGD(d,weighted=True,w=w)
            X = Y.solve(60,t=0.1)
            X = layout_io.normalize_layout(X)

            powers['NP'][a] += get_neighborhood(X,d)
            powers['stress'][a] += get_stress(X,d_norm)

    powers['NP'] /= n
    powers['stress'] /= n

    plt.plot(d_power,powers['stress'],'co-',label="stress")
    plt.plot(d_power,powers['NP'],label="NP")
    plt.suptitle(graph)
    plt.xlabel("d")
    plt.ylabel("")
    plt.legend()
    plt.savefig('figures/update/matrix_power_{}.eps'.format(graph))
    plt.clf()


    return powers


def alpha_exp(n,G,d,d_norm,a,graph):
    alphas = np.linspace(0,1,12)
    alpha = {'NP': np.zeros(len(alphas)), 'stress': np.zeros(len(alphas))}

    k = 10
    w = get_w(G,k=k,a=a)

    for i in range(n):
        for a in range(len(alphas)):
            Y = SGD(d,weighted=True,w=w)
            X = Y.solve(60,t=alphas[a])
            X = layout_io.normalize_layout(X)

            alpha['NP'][a] += get_neighborhood(X,d)
            alpha['stress'][a] += get_stress(X,d_norm)
            print(alpha['stress'][a])


    alpha['NP'] /= n
    alpha['stress'] /= n

    plt.plot(alphas,alpha['stress'],label="stress")
    plt.plot(alphas,alpha['NP'],label="NP")

    plt.legend()
    plt.suptitle(graph)
    plt.savefig('figures/alpha_{}.eps'.format(graph))
    plt.clf()

    return alpha

def epsilon_exp(n,G,d,d_norm,a,graph):
    epsilons = np.linspace(0,0.1,12)
    eps = {'NP': np.zeros(len(epsilons)), 'stress': np.zeros(len(epsilons))}

    k = 21


    for i in range(n):
        for e in range(len(epsilons)):
            w = get_w(G,k=k,a=a,eps = epsilons[e])
            Y = SGD(d,weighted=True,w=w)
            X = Y.solve(60,t=0.1)
            X = layout_io.normalize_layout(X)

            eps['NP'][e] += get_neighborhood(X,d)
            eps['stress'][e] += get_stress(X,d_norm)


    eps['NP'] /= n
    eps['stress'] /= n

    plt.plot(epsilons,eps['stress'],label="stress")
    plt.plot(epsilons,eps['NP'],label="NP")

    plt.legend()
    plt.suptitle(graph)
    plt.savefig('figures/epsilon_{}.eps'.format(graph))
    plt.clf()

    return eps



def experiment(n=5):
    import os
    import pickle
    import copy

    path = 'table_graphs/'
    graph_paths = os.listdir(path)
    #graph_paths = ['block_model300.dot']

    graph_paths = list( map(lambda s: s.split('.')[0], graph_paths) )
    graph_paths = ['netscience','block_model_500']
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
        a=4

        print("Graph: " + graph)
        print("-----------------------------------------------------------")

        print("Iteration experiment")
        #graph_dict['iterations'] = iteration_exp(n,G,d,d_norm,a,graph)
        print("Matrix power experiment")
        #graph_dict['matrix_power'] = matrix_exp(n,G,d,d_norm,graph)
        print("Alpha experiment")
        graph_dict['alpha'] = alpha_exp(n,G,d,d_norm,a,graph)
        print("Epsilon experiment")
        #graph_dict['epsilon'] = epsilon_exp(n,G,d,d_norm,a,graph)

    with open('data/paramater_experiments2.pkl','wb') as myfile:
        pickle.dump(graph_dict,myfile)
    myfile.close()

if __name__ == '__main__':
    experiment(n=15)
