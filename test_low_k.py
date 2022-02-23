import numpy as np
import graph_tool.all as gt
import modules.layout_io as layout_io
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne

from SGD_MDS2 import SGD_MDS2
from SGD_MDS import SGD_MDS

from sklearn.metrics import pairwise_distances

from metrics import get_norm_stress,get_neighborhood

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

def calc_LG(d,d_norm,G,k=8):
    X = SGD_MDS2(d,weighted=True,w=get_w(G,k=k,a=5)).solve(15,debug=True)
    X = layout_io.normalize_layout(X[-1])

    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    gt.graph_draw(G,pos=pos)
    return get_neighborhood(X,d),get_norm_stress(X,d_norm)

def calc_high(d,d_norm,G,k=8):
    X = SGD_MDS(d).solve()
    X = layout_io.normalize_layout(X)

    return get_neighborhood(X,d),get_norm_stress(X,d_norm)


def main(n=5):
    import os
    import pickle
    import copy

    path = 'tsnet-graphs/'
    graph_paths = os.listdir(path)


    template_score = {
        'NP' : [None for i in range(n)],
        'stress': [None for i in range(n)]
    }

    template_alg = {
        'LG_low': copy.deepcopy(template_score),
        'LG_high': copy.deepcopy(template_score)
    }

    scores = {key : copy.deepcopy(template_alg) for key in graph_paths}

    for graph in graph_paths:
        G = gt.load_graph(path+graph)
        d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
        d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

        print("Graph: " + graph)
        print("-----------------------------------------------------------")

        for i in range(n):
            print("Iteration number ", i)

            NP,stress = calc_LG(d,d_norm,G,k=8)
            scores[graph]['LG_low']['NP'][i] = NP
            scores[graph]['LG_low']['stress'][i] = stress
            print("NP: ", NP)

            NP,stress = calc_high(d,d_norm,G)
            scores[graph]['LG_high']['NP'][i] = NP
            scores[graph]['LG_high']['stress'][i] = stress


        with open('data/test_low3.pkl','wb') as myfile:
            pickle.dump(scores,myfile)
        myfile.close()


if __name__ == "__main__":
    main(n=1)
