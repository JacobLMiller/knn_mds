import numpy as np
import graph_tool.all as gt
import modules.layout_io as layout_io
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne

from SGD_MDS2 import SGD_MDS2

from sklearn.metrics import pairwise_distances

def get_neighborhood(X,d,rg = 2):
    """
    How well do the local neighborhoods represent the theoretical neighborhoods?
    Closer to 1 is better.
    Measure of percision: ratio of true positives to true positives+false positives
    """
    norm = np.linalg.norm
    def get_k_embedded(X,k_t):
        dist_mat = pairwise_distances(X)
        return [np.argsort(dist_mat[i])[1:len(k_t[i])+1] for i in range(len(dist_mat))]

    k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0] for i in range(len(d))]

    k_embedded = get_k_embedded(X,k_theory)


    sum = 0
    for i in range(len(X)):
        count_intersect = 0
        intersect = np.intersect1d(k_theory[i],k_embedded[i]).size
        jaccard = intersect / (2*k_theory[i].size - intersect)

        sum += jaccard

    return sum/len(X)

def get_stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(len(X)):
            stress += pow(d[i][j] - np.linalg.norm(X[i]-X[j]),2)
    return stress / np.sum(np.square(d))

def get_w(G,k=5):
    A = gt.adjacency(G).toarray()
    A = np.linalg.matrix_power(A,5)
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

def calc_LG(d,d_norm,G,k=8):
    X = SGD_MDS2(d,weighted=True,w=get_w(G,k=k)).solve()
    return get_neighborhood(X,d),get_stress(X,d_norm)


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

            NP,stress = calc_LG(d,d_norm,G,k=G.num_vertices())
            scores[graph]['LG_high']['NP'][i] = NP
            scores[graph]['LG_high']['stress'][i] = stress


        with open('data/test_low.pkl','wb') as myfile:
            pickle.dump(scores,myfile)
        myfile.close()


if __name__ == "__main__":
    main()
