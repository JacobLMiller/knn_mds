import numpy as np
import graph_tool.all as gt
import scipy.io
import time
import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io
from SGD_MDS import SGD_MDS
from SGD_MDS2 import SGD_MDS2
from MDS_classic import MDS
import modules.graph_io as graph_io

from sklearn.metrics import pairwise_distances


def get_neighborhood(X,d,rg = 1):
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
        sum += count_intersect/(len(k_theory[i]) + len(k_embedded[i]) - count_intersect)

    return sum/len(X)

def stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                stress += pow(d[i][j] - np.linalg.norm(X[i]-X[j]),2)
    return stress / np.sum(np.square(d))

#G = graph_io.load_graph("graphs/dwt_419.vna")
G = gt.lattice([10,40])
#G = gt.load_graph('graphs/block2.dot')
#G = gt.load_graph('graphs/btree8.dot')
d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)

def layout(d):
    Y = SGD_MDS2(d,weighted=False)
    Y.solve()
    print(Y.calc_stress())



def timing(f, n, a):
    print(f.__name__)
    r = range(n)
    t1 = time.perf_counter()
    for i in r:
        f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a)
    t2 = time.perf_counter()
    print((t2-t1)/10)

#timing(layout,1,d)
d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

stress_hist = []
neighbor_hist = []

for k in range(1,20):

    A = gt.adjacency(G).toarray()
    A = np.linalg.matrix_power(A,5)
    A += np.random.normal(scale=0.01,size=A.shape)

    #k = 399
    k_nearest = [np.argpartition(A[i],-k)[-k:] for i in range(len(A))]

    n = G.num_vertices()
    N = 0
    w = np.asarray([[ 1e-5 if i != j else 0 for i in range(len(A))] for j in range(len(A))])
    for i in range(len(A)):
        for j in k_nearest[i]:
            if i != j:
                w[i][j] = 1
                w[j][i] = 1

    for i in range(len(w)):
        for j in range(i):
            if w[i][j] == 1:
                N += 1

    Nc = (n*(n-1))/2 - N
    t = (N/Nc)*np.median(d)


    Y = SGD_MDS2(d,weighted=True,w=w)
    Xs = Y.solve(num_iter=15,t=t,debug=False)

    X = layout_io.normalize_layout(Xs)

    stress_hist.append(stress(X,d_norm))
    neighbor_hist.append(get_neighborhood(X,d))

    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)

    text = G.new_vp('string')
    G.vertex_properties['text']=text

    #Color your edges
    for v in G.vertices():
        text[v] = str(v)

    gt.graph_draw(G,pos=pos,output='drawings/plot/test' + str(2) + '.png')

    # Z = MDS(d,weighted=True,w=w)
    # Z.solve()
    # X = layout_io.normalize_layout(Z.X)
    # print(stress(X,d_norm))
    # print('end')
    # print()
    #
    # pos = G.new_vp('vector<float>')
    # pos.set_2d_array(X.T)

    gt.graph_draw(G,pos=pos,output='drawings/local-global' + str(k) + '.png')


import matplotlib.pyplot as plt
x = np.arange(21)
plt.plot(x,stress_hist,label="Stress")
plt.plot(x,neighbor_hist,label="Neighborhood")
plt.legend()
plt.show()
