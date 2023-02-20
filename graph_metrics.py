import numpy as np
import graph_tool.all as gt

"""
A function (G, X, cluster_ids)

Compute a super-graph, defining distance between distance somehow (adjacecny, weighted based on number of edges, stephen suggestion)

Compute how similar the graph theoretic distnaces and embedded distances are between clusters. 


Load a graph, 
compute clusters,
compute cluster-graph 
compute embedding 
compute quality metric

"""


def get_cluster_ids(G):
    """
    G -> a graph-tool graph 

    returns a list of sets corresponding to cluster ids
    """
    state = gt.minimize_blockmodel_dl(G)
    membership = list(state.get_blocks())
    clusters = np.unique(membership)
    block_map = {c: i for i,c in enumerate(clusters)}
    c_ids = [set() for _ in block_map]
    for v,c in enumerate(membership):
        c_ids[block_map[c]].add(v)
    
    return c_ids

def maybe_add_vertex(G,H,u,v,c_ids):
    if any(G.edge( i,j ) for i in c_ids[u] for j in c_ids[v]):
        H.add_edge(u,v)



def get_cluster_graph(G,c_ids):
    H = gt.Graph(directed=False)
    n,m = len(c_ids), G.num_vertices()
    H.add_vertex(n)
    for u in range(n):
        for v in range(n):
            maybe_add_vertex(G,H,u,v,c_ids)
    
    gt.remove_parallel_edges(H)
    gt.remove_self_loops(H)
    return H
    

def get_neighborhood(G,X):
    d = pairwise_distances(X)
    NE = 0
    for v in G.iter_vertices():
        neighbors = set(G.iter_all_neighbors(v))
        k = len(neighbors)
        top_k = set(np.argsort(d[v])[1:k+1])
        NE += len(top_k.intersection(neighbors))
    return NE / G.num_vertices()

def compute_cluster_metric(G,X,c_ids):
    d = apsp(G)
    low_d_clusters = find_cluster_centers(X,c_ids)

    return get_neighborhood(G,low_d_clusters)



G = gt.load_graph("graphs/connected_watts_400.dot")

c_ids = get_cluster_ids(G)

H = get_cluster_graph(G,c_ids)


gt.graph_draw(H)