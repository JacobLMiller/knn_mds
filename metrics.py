import numpy as np
from numba import jit

from sklearn.metrics import pairwise_distances

@jit(nopython=True)
def get_norm_stress(X,d):
    norm, stress, n = np.linalg.norm, 0, len(X)

    for i in range(n):
        for j in range(n):
            stress += pow(d[i][j] - norm(X[i]-X[j]),2)
    return stress / np.sum(np.square(d))

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


    NP = 0
    for i in range(len(X)):
        intersect = np.intersect1d(k_theory[i],k_embedded[i]).size
        jaccard = intersect / (2*k_theory[i].size - intersect)

        NP += 1-jaccard

    return NP / len(X)




def ind_neighborhood(X,d,rg = 2):
    """
    How well do the local neighborhoods represent the theoretical neighborhoods?
    Closer to 1 is better.
    Measure of percision: ratio of true positives to true positives+false positives
    """
    norm = np.linalg.norm
    def get_k_embedded(X,k_t):
        dist_mat = pairwise_distances(X)
        return [np.argpartition(dist_mat[i],len(k_t[i]))[:len(k_t[i])] for i in range(len(dist_mat))]

    k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0] for i in range(len(d))]

    k_embedded = get_k_embedded(X,k_theory)

    sum = 0
    for i in range(len(X)):
        count_intersect = 0
        for j in range(len(k_theory[i])):
            if k_theory[i][j] in k_embedded[i]:
                count_intersect += 1
        sum += count_intersect/ len(k_theory[i])
        yield 1- count_intersect/len(k_theory[i])

    return


def MAP(G,X):
    V = G.num_vertices()

    embed_dist = pairwise_distances(X)
    R = np.array([np.argsort(embed_dist[i])[1:] for i in range(V)])

    outer_sum = 0
    for a in G.vertices():
        Na = np.array([int(v) for v in a.out_neighbors()])
        v = int(a)

        inner_sum = 0
        for i in range(Na.size):
            b_i = R[v].tolist().index(Na[i])
            Ra_bi = R[v][:b_i+1]

            inner_sum += np.intersect1d(Na,Ra_bi).size / Ra_bi.size

        outer_sum += inner_sum/Na.size

    return outer_sum / V

def KL_div(X,d,s):
    p = lambda d: np.exp(-pow(d,2)/s)
    q = lambda m: np.exp(-pow(np.linalg.norm(m),2)/s)
    n = len(d)

    P = np.array([[p(d[i][j]) if i != j else 0 for j in range(n)]
            for i in range(n)])

    for i in range(len(P)):
        P[i] = P[i] / np.linalg.norm(P[i])

    Q = np.array([[q(X[i]-X[j]) if i != j else 0 for j in range(n)]
            for i in range(n)])

    for i in range(len(Q)):
        Q[i] = Q[i] / np.linalg.norm(Q[i])



    KL = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                p_ij = P[i][j]
                q_ij = Q[i][j]

                KL += p_ij*np.log(p_ij/q_ij) + q_ij*np.log(q_ij/p_ij)
    return KL/2



def chen_neighborhood(D,X,k):
    embed_dist = pairwise_distances(X)

    percision = 0
    for i in range(len(D)):
        embed_k = np.argsort(embed_dist[i])[1:k+1]


        dK = np.argsort(D[i])[1:k+1][-1]
        dK = D[i][dK]

        count = 0
        for j in embed_k:
            if D[i][j] <= dK:
                count += 1
        percision += (count/k)

    return percision / len(D)

def avg_lcl_err(X,D):
    embed = pairwise_distances(X)

    n = len(D)
    err = [0 for _ in range(n)]
    for i in range(n):
        max_theory = np.max(D[i])
        max_embed = np.max(embed[i])

        local = 0
        for j in range(n):
            if i != j:
                local += abs(D[i][j]/max_theory - embed[i][j]/max_embed)
        err[i] = local / (n-1)
    return err
