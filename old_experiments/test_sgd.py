import numpy as np
import graph_tool.all as gt
import modules.layout_io as layout_io
import modules.distance_matrix as distance_matrix

from SGD_MDS import SGD_MDS

from metrics import get_stress,get_neighborhood

import matplotlib.pyplot as plt
import modules.thesne as thesne
import s_gd2


def draw(G,X,output=None):
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    #
    if output:
        gt.graph_draw(G,pos=pos,output=output)
    else:
        gt.graph_draw(G,pos=pos)

def run_tsnet(name,G,d,d_norm):
    lambdas_2 = [1,1.2,0]
    lambdas_3 = [1,0.01,0.6]
    X,hist = thesne.tsnet(
            d_norm, output_dims=2, random_state=1, perplexity=40, n_epochs=2000,
            Y=None,
            initial_lr=50, final_lr=50, lr_switch=2000 // 2,
            initial_momentum=0.5, final_momentum=0.5, momentum_switch=2000 // 2,
            initial_l_kl=lambdas_2[0], final_l_kl=lambdas_3[0], l_kl_switch=2000 // 2,
            initial_l_c=lambdas_2[1], final_l_c=lambdas_3[1], l_c_switch=2000 // 2,
            initial_l_r=lambdas_2[2], final_l_r=lambdas_3[2], l_r_switch=2000 // 2,
            r_eps=0.05, autostop=1e-7, window_size=40,
            verbose=True, a=1)
    X = layout_io.normalize_layout(X)

    draw(G,X,output='drawings/tsnet_compare/{}.png'.format(name))

    return get_neighborhood(X,d),get_stress(X,d_norm)

def run_sgd(name,G,d,d_norm):
    I = []
    J = []
    for e1,e2 in G.iter_edges():
        I.append(e1)
        J.append(e2)

    X = s_gd2.layout_convergent(I, J)
    X = layout_io.normalize_layout(X)

    draw(G,X,output='drawings/sgd_compare/{}.png'.format(name))

    return get_neighborhood(X,d),get_stress(X,d_norm)

def experiment(n=5):
    import os
    import pickle
    import copy

    path = 'table_graphs/'
    graph_paths = os.listdir(path)

    graph_paths = list( map(lambda s: s.split('.')[0], graph_paths) )
    #graph_paths = ['custom_cluster_100']
    print(graph_paths)

    adjacency_len = len( np.linspace(5,100,8) )
    radius_len = len( list(range(1,8+1) ))
    linear_len = len( np.linspace(0,1,8) )

    zeros = lambda s: np.zeros(s)
    alg_dict = {'NP': 0, 'stress': 0}


    graph_dict = {key: copy.deepcopy(alg_dict) for key in graph_paths}

    for graph in graph_paths:
        print(graph)
        G = gt.load_graph(path+graph + '.dot')
        d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
        d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)
        if G.num_vertices() <= 1010: continue

        CC,_ = gt.global_clustering(G)
        a = 2 if CC < 0.1 else 3 if CC < 0.4 else 4 if CC < 0.6 else 5

        print("Graph: " + graph)
        print("-----------------------------------------------------------")

        for i in range(n):
            print()
            print("Iteration number ", i)

            NP,stress = run_tsnet(graph,G,d,d_norm)
            graph_dict[graph]['NP'] += NP
            graph_dict[graph]['stress'] += stress


        graph_dict[graph]['NP'] /= n
        graph_dict[graph]['stress'] /= n

        print()
        print()

    with open('data/tsnet_table_graphs_large.pkl','wb') as myfile:
        pickle.dump(graph_dict,myfile)
    myfile.close()

experiment(n=5)
