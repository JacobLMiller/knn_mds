from SGD_MDS2 import SGD_MDS2
from SGD_MDS import SGD_MDS
from MDS_classic import MDS
import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import numpy as np
import graph_tool.all as gt
import scipy.io

from tsnet_repeat import get_neighborhood,get_stress
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
        yield 1-jaccard

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

    print(KL/2)
    return KL/2


def KL_2(X,d,k):
    n = len(X)
    sig = np.log(k)

    q_denom = sum( pow(1+pow(np.linalg.norm(X[i]-X[j]),2),-1) for j in range(n) for i in range(n) if i !=j)

    pi_given_j = lambda i,j: np.exp(-pow(d[i][j],2)/(2*pow(sig,2))) / sum([np.exp(-pow(d[i][k],2) / (2*pow(sig,2))) for k in range(n) if i != k]) if i != j else 0

    pij = lambda i,j: (pi_given_j(i,j)+pi_given_j(j,i))/(2*n) if i != j else 0

    qij = lambda i,j: pow(1+pow(np.linalg.norm(X[i]-X[j]),2),-1) / q_denom if i != j else 0



    C = 0
    for i in range(n):
        for j in range(n):
            if i!=j:
                C += pij(i,j)*np.log(pij(i,j)/qij(i,j))
    print(C)
    return C



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
    return np.array(err)


def my_random_graph(n,b,edge_probs):
    G = gt.Graph(directed=False)
    G.add_vertex(n)

    bm = G.new_vp('int')
    G.vertex_properties['bm'] = bm

    for v in G.vertices():
        bm[v] = random.randint(0,b-1)

    for i in range(G.num_vertices()):
        for j in range(i):
            if random.random() < edge_probs(bm[i],bm[j]):
                G.add_edge(i,j)
    return G,bm

def prob(a, b):

   if a == b:

       return 0.3

   else:

       return 0.001
import random
# G, bm = gt.random_graph(400, lambda: np.random.poisson(10), directed=False,
#
#                         model="blockmodel",
#
#                         block_membership=lambda: random.randint(0,4),
#
#                         edge_probs=prob)

G,bm = my_random_graph(100,4,prob)
#G.save('graphs/dummyblock.dot')

# G = gt.Graph(directed=False)
# G.add_vertex(3)
# G.add_edge_list([(0,1),(1,2),(2,0)])

#G = gt.generate_sbm(list(bm), probs, out_degs=None, directed=False, micro_ers=False, micro_degs=False)


#G = gt.load_graph("graphs/dwt_419.dot")
G = gt.lattice([10,10])
G = gt.load_graph('graphs/dwt_1005.dot')
# G = gt.load_graph('graphs/btree8.dot')
d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)

def get_w(k=5,a=5):
    A = gt.adjacency(G).toarray()
    A = np.linalg.matrix_power(A,a)
    A += np.random.normal(scale=0.01,size=A.shape)

    #k = 10
    k_nearest = [np.argpartition(A[i],-k)[-k:] for i in range(len(A))]

    n = G.num_vertices()
    N = 0
    w = np.asarray([[ 0 if i != j else 0 for i in range(len(A))] for j in range(len(A))])
    for i in range(len(A)):
        for j in k_nearest[i]:
            if i != j:
                w[i][j] = 1
                w[j][i] = 1


    return w

# for i in range(len(w)):
#     for j in range(i):
#         if w[i][j] == 1:
#             N += 1

# Nc = (n*(n-1))/2 - N
#t = (N/Nc)*np.median(d)*0.1
t = 0.7
T = np.linspace(0,1,10)
def power(n,count):
    for i in range(count):
        yield pow(n,i)



# for k in K:
#     k = k if k < G.num_vertices() else G.num_vertices()
#     w = get_w(k)
#     for t in T:
d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True,verbose=False)

NP = []
stress = []
NH = []
map = []
lcl = []
print(G.num_vertices())


k = 8
a = int(np.max(d))

print(np.max(d))
for a in [2]:
    w = get_w(k=8,a=a)
    #w = gt.adjacency(G).toarray()

    count = 0
    temp_stress = 0

    t = np.count_nonzero(w)/w.size
    Y = SGD_MDS2(d,weighted=True,w=w)
    Xs = Y.solve(num_iter=100,t=t,debug=True)
    Zx = Xs[-1]

    X = layout_io.normalize_layout(Zx)
    classic_nei = np.array([x for x in get_neighborhood(Zx,d)])
    print(np.mean(classic_nei))
    print(get_stress(X,d_norm))

    pos = G.new_vp('vector<float>')
    pos.set_2d_array(Zx.T)
    #
    gt.graph_draw(G,pos=pos)


    for layout in Xs:
        X = layout_io.normalize_layout(layout)

        stress.append(get_stress(X,d_norm))
        pos = G.new_vp('vector<float>')
        pos.set_2d_array(X.T)
        #
        gt.graph_draw(G,pos=pos,output='drawings/test/dwt_k_' + str(k) + '_a' + str(a) + '_' +str(count) + '.png')
        count += 1
    print(a)
    print(k)
    nei = np.array([x for x in get_neighborhood(X,d)])
    print("Avg NP score:", nei.mean())
    print("Compared to ", np.mean(classic_nei) )
    print()
