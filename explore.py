from SGD_MDS import SGD_MDS, k_nearest_embedded
from MDS_classic import MDS
import graph_tool.all as gt
import time
import numpy as np
import matplotlib.pyplot as plt

import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne
import scipy.io
import random
import pickle

norm = lambda x: np.linalg.norm(x,ord=2)


def get_neighborhood(X,d,rg = 1):
    """
    How well do the local neighborhoods represent the theoretical neighborhoods?
    Closer to 1 is better.
    Measure of percision: ratio of true positives to true positives+false positives
    """
    def get_k_embedded(X,k_t):
        dist_mat = [[norm(X[i]-X[j]) if i != j else 10000 for j in range(len(X))] for i in range(len(X))]
        return [np.argpartition(dist_mat[i],len(k_t[i]))[:len(k_t[i])] for i in range(len(dist_mat))]

    k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0] for i in range(len(d))]
    k_embedded = get_k_embedded(X,k_theory)

    sum = 0
    for i in range(len(X)):
        count_intersect = 0
        for j in range(len(k_theory[i])):
            if k_theory[i][j] in k_embedded[i]:
                count_intersect += 1
        sum += count_intersect/(len(k_theory[i])+len(k_embedded[i])-count_intersect)

    return sum/len(X)

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

def get_distortion(X,d):
    dist = 0
    for i in range(len(X)):
        for j in range(i):
            dist += abs(norm(X[i]-X[j]) - d[i][j])/d[i][j]
    return dist/choose(len(X),2)



def deg():
    return 2
def prob(a, b):

   if a == b:

       return 0.999

   else:

       return 0.001
def main():
    G = graph_io.load_graph('graphs/block2.dot')
    # G = gt.random_graph(100,deg,directed=False)

    #
    # G, bm = gt.random_graph(100, lambda: np.random.poisson(lam=10), directed=False,
    #
    #                         model="blockmodel",
    #
    #                         block_membership=lambda: random.randint(0,2),
    #
    #                         edge_probs=prob)
    #
    # G.save("graphs/block2.dot")

    #G = graph_io.load_graph('graphs/dwt_419.vna')
    # print(G)
    # for v in G.vertices():
    #     print(v)
    #
    #G = gt.lattice([40,10])
    K = np.array([i for i in range(2,100,5)])
    data = [None,None]


    #
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

    A = gt.adjacency(G).toarray()
    A = np.linalg.matrix_power(A,5)
    A += np.random.normal(scale=0.01,size=A.shape)
    print(A)

    k = 40
    k_nearest = [np.argpartition(A[i],-k)[-k:] for i in range(len(A))]

    w = np.asarray([[ 1e-5 if i != j else 0 for i in range(len(A))] for j in range(len(A))])
    for i in range(len(A)):
        for j in k_nearest[i]:
            if i != j:
                w[i][j] = 1
                w[j][i] = 1
    print(w)

    Y = SGD_MDS(d,weighted=True,k=2,w=w)
    Y.solve(15,debug=True)
    Z = layout_io.normalize_layout(Y.X)
    print("Distortion: ", get_distortion(Z,d_norm))
    print("Neighborhoood: ", get_neighborhood(Y.X,d))

    # print(get_neighborhood(Y.X,d,rg=1))
    # #print(Y.d == d)
    #
    # vertex_color = G.new_vp('vector<double>')
    # G.vertex_properties['vertex_color']=vertex_color
    #
    # #Color your edges
    # for v in G.vertices():
    #     if v < 20:
    #         vertex_color[v] = (0.0, 255.0, 0.0, 1)
    #     elif v < 40:
    #         vertex_color[v] = (255.0, 0.0, 0.0, 1)
    #     elif v < 60:
    #         vertex_color[v] = (0.0, 0.0, 255.0, 1)
    #     elif v < 80:
    #         vertex_color[v] = (120.0, 120.0, 0.0, 1)
    #     else:
    #         vertex_color[v] = (0.0, 120.0, 120.0, 1)

    pos = G.new_vp('vector<float>')
    pos.set_2d_array(Z.T)

    gt.graph_draw(G,pos=pos)
    #
    adj = w
    H = gt.Graph(directed=False)
    H.add_vertex(len(adj))
    H.add_edge_list(np.argwhere(adj >= 0.5))
    gt.remove_parallel_edges(H)

    pos = H.new_vp('vector<float>')
    pos.set_2d_array(Y.X.T)

    gt.graph_draw(H,pos=pos)

    # x = np.arange(len(Y.neighbor_hist))
    # plt.plot(x,Y.neighbor_hist)
    # plt.ylim(0,1)
    # plt.show()

def draw_increase_k():
    distortions = []
    neighbors = []
    #
    for k in K:

        k_nearest = [np.argpartition(A[i],-k)[-k:] for i in range(len(A))]

        w = np.asarray([[ 1e-7 if i != j else 0 for i in range(len(A))] for j in range(len(A))])
        for i in range(len(A)):
            for j in k_nearest[i]:
                if i != j:
                    w[i][j] = 1
                    w[j][i] = 1

        Y = SGD_MDS(d,weighted=True,k=k,w=w)
        #print(Y.w)
        print("k: ", k)
        Y.solve(15,debug=False)
        Z = layout_io.normalize_layout(Y.X)


        distortions.append(get_distortion(Z,d_norm))
        neighbors.append(1-get_neighborhood(Y.X,d))

        pos = G.new_vp('vector<float>')
        pos.set_2d_array(Z.T)

        gt.graph_draw(G,pos=pos,output='drawings/k-trials' + str(k) + '.png')
    #
    #     adj = Y.w.copy()
    #     H = gt.Graph(directed=False)
    #     H.add_vertex(len(adj))
    #     H.add_edge_list(np.argwhere(adj >= 0.5))
    #     gt.remove_parallel_edges(H)
    #     print('k: ',k)
    #     dist = get_distortion(Y.X,d/1000)
    #     distortions.append(min(dist,1))
    #     neighbors.append(1-get_neighborhood(Y.X,d))
    #     print("distortion: ", distortions[-1])
    #     print("neighborhood: ", neighbors[-1])
    #
    #     # pos = H.new_vp('vector<float>')
    #     # pos.set_2d_array(Y.X.T)
    #     #
    #     # gt.graph_draw(H,pos=pos,output='figures/overlays/test_' + str(k) + 'under.png')
    #
    #     pos = G.new_vp('vector<float>')
    #     pos.set_2d_array(Y.X.T)
    #
    #     gt.graph_draw(G,pos=pos,output='figures/overlays/test_mesh_' + str(k) + 'full.png')
    #
    x = np.arange(len(K))
    plt.xticks(ticks=np.arange(len(K)),labels=K.astype('str'))
    plt.plot(x,distortions,'o-',label="Distortion")
    plt.plot(x,neighbors,'o-',label="Neighborhoood")
    plt.ylim(0,1)
    plt.legend()
    plt.show()

    for i in range(len(Y.hist)):
        pos = G.new_vp('vector<float>')
        pos.set_2d_array(layout_io.normalize_layout(Y.hist[i]).T)

        #gt.graph_draw(H,pos=pos,output='figures/overlays/test_k' + str(i) + '.png')
        gt.graph_draw(G,pos=pos,output='drawings/test2/test_k' + str(i) + '.png')

    # x = np.arange(len(Y.stress_hist))
    # plt.plot(x,Y.stress_hist)
    # plt.show()
    # plt.plot(x,Y.neighbor_hist)
    # plt.show()

    # A = scipy.io.mmread('graphs/fpga_dcop_08.mtx').toarray()
    # G = gt.Graph(directed=False)
    # G.add_edge_list(np.transpose(A.nonzero()))
    # gt.remove_parallel_edges(G)
    # gt.remove_self_loops(G)
    # G.save('fpga.dot')

if __name__ == "__main__":
    #main()

    import os

    directory = 'graphs'
    for filename in os.scandir(directory):
        if filename.is_file():
            print(filename.path)
            G = graph_io.load_graph(filename.path)
            newpath = 'dot-graphs/' + filename.path.split('/')[1].split('.')[0] + '.dot'
            G.save(newpath,fmt='dot')
            #
    print(1000*1e-5)
