import numpy as np
import graph_tool.all as gt

from SGD_MDS2 import SGD
from SGD_MDS import SGD_MDS
import modules.distance_matrix as distance_matrix

from metrics import get_stress,get_neighborhood,get_cost

import matplotlib.pyplot as plt

import gc

import time

from simple_driver import get_w


def time_exp(n=5):
    import os
    path = 'new_tests/'
    graph_paths = os.listdir(path)

    graph_paths = list( map(lambda s: s.split('.')[0], graph_paths) )
    #graph_paths = ['connected_watts_300']

    graphs = [gt.load_graph('new_tests/{}.dot'.format(graph)) for graph in graph_paths]
    details = [ (g.num_vertices(), g.num_edges(), name) for name,g in zip(graph_paths,graphs)]
    details.sort()

    data = {}


    for _,__,graph in details:
        G = gt.load_graph('new_tests/{}.dot'.format(graph))
        avg_time = []

        for i in range(n):

            start = time.perf_counter()
            d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)

            w = get_w(G,k=10,a=5)
            Y = SGD(d,weighted=True, w = w)
            X = Y.solve(60,t=0.1)
            end = time.perf_counter()
            print('took: {}'.format(end-start))
            avg_time.append(end-start)

        data[graph] = sum(avg_time)/len(avg_time)
        print(data[graph])

    return data


def draw(G,X,output=None):
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    #
    if output:
        gt.graph_draw(G,pos=pos,output=output)
    else:
        gt.graph_draw(G,pos=pos)

def cost_curve(n=5):
    import os

    path = 'example_graphs/'
    graph_paths = os.listdir(path)

    graph_paths = list( map(lambda s: s.split('.')[0], graph_paths) )
    graph_paths = ['powerlaw300','block_model300']
    print(graph_paths)
    graph_costs = {}
    for graph in graph_paths:
        G = gt.load_graph(path+graph + '.dot')
        d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
        cost_array = np.zeros(61)
        print(graph)
        for i in range(n):
            w = get_w(G,k=200,a=5)
            Y = SGD(d,weighted=True, w = w)
            X = Y.solve(60,t=0.1,debug=True,eps=0.01)
            # if i == 0:
            #     for j in range(len(X)):
            #         if j % 10 == 0:
            #             draw(G,X[j],output='drawings/update/test_{}.png'.format(j))
            #
            cost_array += np.array( [get_cost(x,d,w,0.1) for x in X] )

        cost_array /= n

        plt.plot(np.arange(61),cost_array)
        plt.suptitle(graph)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig(graph + "cost.png")
        plt.close()
        graph_costs[graph] = cost_array

    return graph_costs


timing = time_exp(n=5)
curves = cost_curve(5)
curves = 0

data = {'timing': timing, 'curves': curves}

import pickle
with open('data/extra_exps.pkl','wb') as myfile:
    pickle.dump(data,myfile)
myfile.close()
