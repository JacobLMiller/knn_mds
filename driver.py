from modules.L2G import L2G, get_w
import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import matplotlib.pyplot as plt
import numpy as np
import graph_tool.all as gt

from metrics import get_neighborhood, get_stress,get_cost
from sklearn.metrics import pairwise_distances

from graph_metrics import apsp

def compute_umap(G):
    d = apsp(G)
    from umap import UMAP
    X = UMAP(10,metric="precomputed").fit_transform(d)

    draw(G,X,output="umap_draw.png")


def layout(G,d,d_norm,debug=True, k=7, a=5,t=0.6,radius=False):
    k = k if k < G.num_vertices() else G.num_vertices()
    w = get_w(G,k=k,a=a)
    Y = SGD(d,weighted=True,w=w)
    Xs = Y.solve(60,debug=debug,radius=radius,t=t)
    X = layout_io.normalize_layout(Xs)

    return X,w


def draw(G,X,output=None):
    from graph_tool import GraphView

    b = 50

    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    #
    u = GraphView(G, efilt=lambda e: True)
    deg = G.degree_property_map("total")
    deg.a = 4 * (np.sqrt(deg.a) * 0.5 + 0.4)


    if output:
        gt.graph_draw(u,pos=pos,output=output)#,vertex_fill_color=color)
    else:
        gt.graph_draw(G,pos=pos)





def diffusion_weights(d,a=5, k = 20, sigma=1):
    #Transform distance matrix
    diff = np.exp( -(d**2) / (sigma **2) )
    diff /= np.sum(diff,axis=0)

    #Sum powers from 1 to a
    mp = np.linalg.matrix_power
    A = sum( mp(diff,i) for i in range(1,a+1) )
    A = mp(diff,a)

    # A = (pow(2,10*A) - 1)

    #Find k largest points for each row 
    Neighbors = set()
    for i in range(diff.shape[0]):
        args = np.argsort(A[i])[::-1][1:k+1]
        for j in args:
            Neighbors.add( (int(i),int(j)) )

    #Set pairs to 1
    w = np.zeros_like(diff)
    for i,j in Neighbors:
        w[i,j] = 1
        w[j,i] = 1
    return w,A

from sklearn.metrics import pairwise_distances
import time

def sample_k(max):

    accept = False

    while not accept:

        k = np.random.randint(1,max+1)

        accept = np.random.random() < 1.0/k

    return k

def get_graph(n=200):
    return gt.random_graph(n, lambda: sample_k(40), model="probabilistic-configuration",

                    edge_probs=lambda i, k: 1.0 / (1 + abs(i - k)), directed=False,

                    n_iter=100)

def measure_time(repeat=5):
    sizes = list(range(200,4005,200))
    print(len(sizes))
    times = np.zeros(len(sizes))

    for i,n in enumerate(sizes):
        print(f"On the {n}th size")
        for _ in range(repeat):
            G = get_graph(n)
            start = time.perf_counter() 
            d = distance_matrix.get_distance_matrix(G,"spdm")
            w = get_w(G,a=10,k=35)
            X = L2G(d,weighted=True,w=w).solve(200)
            end = time.perf_counter()

            val = end-start
            times[i] += val

        times[i] /= repeat
        np.savetxt("03_13_timeexp.txt",times)


def drive(graph,hist=False,output=None,k=10):
    # G = gt.load_graph("{}".format(graph))
    G = graph
    print(f"|V|: {G.num_vertices()}, |E|: {G.num_edges()}")

    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)
    print(d)

    # w,A = diffusion_weights(d,k=k,a=2)
    # metric = lambda pi, pj: np.sqrt(np.linalg.norm(np.log(pi) - np.log(pj)))
    # B = pairwise_distances(A)
    # print(B.shape)

    w = get_w(G,a=10,k=5)
    print(w)

    start = time.perf_counter()
    Y = L2G(d,weighted=True, w = w)

    X = Y.solve(200,debug=hist,t=0.1,log=True)
    print(f"Optimization took {time.perf_counter()-start}s")
    X = layout_io.normalize_layout(X)

    print('NP: {}'.format(get_neighborhood(X,d,1)))
    print('stress: {}'.format(get_stress(X,d)))

    if hist:
        draw_hist(G,X,d,w,Y)
    else:
        draw(G,X,output=output)

if __name__ == '__main__':
    G = gt.load_graph("graphs/connected_watts_500.dot")
    compute_umap(G)

