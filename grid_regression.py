import numpy as np
import graph_tool.all as gt
import modules.layout_io as layout_io
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne

from SGD_MDS2 import SGD_MDS2
from SGD_MDS import SGD_MDS

from sklearn.metrics import pairwise_distances

from metrics import get_norm_stress,get_neighborhood

import random

def get_w(G,k=5,a=1):
    A = gt.adjacency(G).toarray()
    mp = np.linalg.matrix_power
    A = np.array(sum([mp(A,i) for i in range(1,a+1)]))
    A += np.random.normal(scale=0.01,size=A.shape)


    #k = 10
    k_nearest = [np.argpartition(A[i],-k)[-k:] for i in range(len(A))]

    n = G.num_vertices()
    w = np.asarray([[ 0 if i != j else 0 for i in range(len(A))] for j in range(len(A))])
    for i in range(len(A)):
        for j in k_nearest[i]:
            if i != j:
                w[i][j] = 1
                w[j][i] = 1

    return w

def calc_LG(d,d_norm,G,k=8,a=1):
    k = k if k < G.num_vertices() else G.num_vertices()
    X = SGD_MDS2(d,weighted=True,w=get_w(G,k=k,a=a)).solve(60,debug=True)
    X = layout_io.normalize_layout(X[-1])
    return get_neighborhood(X,d),get_norm_stress(X,d_norm),X

def calc_high(d,d_norm,name,G,k=8):
    X = SGD_MDS(d).solve()
    X = layout_io.normalize_layout(X)

    return get_neighborhood(X,d),get_norm_stress(X,d_norm)


def get_block(n=100):
    def prob(a, b):

       if a == b:

           return 0.999

       else:

           return 0.001

    G, bm = gt.random_graph(n, lambda: np.random.poisson(lam=10), directed=False,

                            model="blockmodel",

                            block_membership=lambda: random.randint(0,4),

                            edge_probs=prob)
    return G

def main(n=5):
    import os
    import pickle
    import copy


    path = 'tsnet-graphs/'
    graph_paths = os.listdir(path)
    graph_paths = ['grids', 'long_grids']
                #'blocks']


    template_score = {
        'NP' : {},
        'stress': {}
    }

    template_alg = {
        'grids': copy.deepcopy(template_score),
        'long_grids': copy.deepcopy(template_score)
        #'blocks': copy.deepcopy(template_score)
    }

    scores = {key : copy.deepcopy(template_score) for key in graph_paths}

    params = {'a': [i for i in range(1,30)],
              'k': [2,4,6,8,12,18,24] + [i for i in range(50,401,50)],
              }

    graphs = {
        'grids': [gt.lattice([i,i]) for i in range(4,20)],
        'long_grids': [gt.lattice([i,2*i]) for i in range(4,10)]
        # 'blocks': [get_block(n=i) for i in range(10,400,10)]
    }


    for graph_type in scores.keys():

        for G in graphs[graph_type]:
            d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
            d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)
            print()
            print('----------------------------------------------------------------')
            print()
            print(graph_type)

            for a in params['a']:
                print('a:', a)
                print()

                for k in params['k']:
                    print('k:', k)
                    print()

                    total_NP,total_stress = 0,0
                    for i in range(n):

                        NP,stress,X = calc_LG(d,d_norm,G,k=k,a=a)

                        total_NP += NP
                        total_stress += stress

                    pos = G.new_vp('vector<float>')
                    pos.set_2d_array(X.T)
                    gt.graph_draw(G,pos=pos,output='drawings/grids2/' + graph_type + '_a' + str(a) + '_k' + str(k) + '_i' + str(i) + '.png')

                    scores[graph_type]['NP']['V' + str(G.num_vertices()) + '_a' + str(a) + '_k' + str(k)] = NP/n
                    scores[graph_type]['stress']['V' + str(G.num_vertices()) + '_a' + str(a) + '_k' + str(k)] = stress/n

                    with open('data/grid_regression2.pkl','wb') as myfile:
                        pickle.dump(scores,myfile)
                    myfile.close()





if __name__ == "__main__":
    main()
