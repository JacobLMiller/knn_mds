import graph_tool.all as gt
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import itertools


def social_model(n=100,p=0.7):
    candidates = set(itertools.combinations(range(n), 2))
    G = gt.Graph(directed=False)
    G.add_vertex(n=n)

    while len(candidates) > 0:
        print(len(candidates))
        consider = candidates.pop()
        u,x = consider
        if random.random() > 0.5: u,x = x,u

        C = {x}
        for a in G.iter_all_neighbors(u):
            for b in G.iter_all_neighbors(a):
                if b != u and b not in C and u not in set(G.iter_all_neighbors(b)): C.add(b)
        v = C.pop()

        if random.random() <= p:
            G.add_edge(u,v)

            candidates.discard( (u,v) )
            candidates.discard( (v,u) )


    return G

def convert_graph(H):
    H = nx.convert_node_labels_to_integers(H)
    G = gt.Graph(directed=False)
    G.add_vertex(n=len(H.nodes()))
    for e in H.edges():
        G.add_edge(e[0],e[1])
    return G

def block_model(n=500,lam=10,num_blocks=5):
    def prob(a, b):

       if a == b:

           return 0.999

       else:

           return 0.001
    G, bm = gt.random_graph(n, lambda: np.random.poisson(lam=lam), directed=False,

                            model="blockmodel",

                            block_membership=lambda: random.randint(0,num_blocks-1),

                            edge_probs=prob)
    return G,bm

# BA = lambda n,m,c: gt.price_network(n,m=m,c=c,directed=False)
#
# for _ in range(100):
#     c = random.random()
#     m = random.randint(1,10)
#     G = BA(1000,m,c)
#     cc,_ = gt.global_clustering(G)
#     print(cc)

def custom_cluster(n=100,k=5,p_in=0.99,p_out=0.01):
    #Normalize so probs sum to 1
    # total = p_in + p_out
    # p_in, p_out = p_in/total, p_out/total

    #Instantiate graph
    G = gt.Graph(directed=False)
    G.add_vertex(n=n)

    #Assign clusters
    clust_assign = [random.randint(0,k-1) for _ in range(n)]

    #Add edges with prob based on cluster assignment
    for i in range(n):
        for j in range(i):
            chance = random.random()
            if clust_assign[i] == clust_assign[j] and chance < p_in:
                G.add_edge(i,j)
            elif chance < p_out:
                G.add_edge(i,j)

    return G


# G = social_model(p=0.1)
# clust = gt.local_clustering(G)
# print(gt.vertex_average(G,clust))
# G.save('graphs/social_test.dot')

# G = custom_cluster(n=250,p_in=0.5,p_out=0.001)
# clust = gt.local_clustering(G)
# print(gt.vertex_average(G,clust))
# G.save('graphs/custom_cluster250.dot')
# print(type( set(G.iter_all_neighbors(4)) ))
#
# import pickle
# with open('mnist_8_3.pkl', 'rb') as myfile:
#     data = pickle.load(myfile)
#
# G,w = gt.generate_knn(data,k=3)
# clust = gt.local_clustering(G)
# print(gt.vertex_average(G,clust))
# G.save('graphs/test_mnist.dot')

import os
graph_paths = os.listdir('graphs/')

# for graph in graph_paths:
#     G = gt.load_graph('graphs/{}'.format(graph))
#     clust = gt.local_clustering(G)
#     print("Graph: {}, CC: {}".format(graph, gt.vertex_average(G,clust)) )

for i in [100,200,300,400,500,700,1000,1500]:
    G,bm = block_model(n=i)
    CC = gt.local_clustering(G)
    G.vertex_properties['block'] = bm

    G.save('random_runs/block_model{}.dot'.format(i))



# for n in [200]:
#     print(n)
#     bad = True
#     while bad:
#         k = random.randint(2,30)
#         p = random.random()
#         print("K: {}, p: {}".format(k,p))
#         #G = convert_graph( nx.Graph(nx.connected_watts_strogatz_graph(n, k, random.random()) ))
#         #G = custom_cluster(n=n,k=k)
#         G = convert_graph( nx.Graph( nx.powerlaw_cluster_graph(n,k,p=p) ) )
#         clust = gt.local_clustering(G)
#         CC,_ = gt.vertex_average(G,clust)
#         print(CC)
#         if CC > 0.6:
#             G.save('random_runs/powerlaw{}.dot'.format(n))
#             bad = False

G = gt.load_graph('random_runs/powerlaw300.dot')
clust = gt.local_clustering(G)
print("Graph: {}, CC: {}".format("powerlaw300", gt.vertex_average(G,clust)) )

G = gt.load_graph('random_runs/connected_watts_300.dot')
clust = gt.local_clustering(G)
print("Graph: {}, CC: {}".format("connected_watts_300", gt.vertex_average(G,clust)) )

import gc

print(gc.garbage)
