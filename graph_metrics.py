import numpy as np
import graph_tool.all as gt
from sklearn.metrics import pairwise_distances
from metrics import find_cluster_centers

cmap = {
    0: "red",
    1: "blue",
    2: "green",
    3: "orange",
    4: "brown",
    5: "purple",
    6: "yellow",
    7: "grey",
    8: "cyan",
    9: "black",
    10: "white"
}

cmap = {i: cmap[i%10] for i in range(100)}

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

def apsp(G,weights=None):
    d = np.array( [v for v in gt.shortest_distance(G,weights=weights)] ,dtype=float)
    return d


def get_cluster_ids(G: gt.Graph):
    """
    G -> a graph-tool graph 

    returns a list of sets corresponding to cluster ids
    """
    state = gt.minimize_blockmodel_dl(G)
    for _ in range(5):
        tmp_state = gt.minimize_blockmodel_dl(G)
        if tmp_state.entropy() < state.entropy():
            state = tmp_state
    membership = list(state.get_blocks())
    clusters = np.unique(membership)
    block_map = {c: i for i,c in enumerate(clusters)}
    c_ids = [set() for _ in block_map]
    for v,c in enumerate(membership):
        c_ids[block_map[c]].add(v)
    
    return c_ids,state

def maybe_add_vertex(G,H,u,v,c_ids):
    if any(G.edge( i,j ) for i in c_ids[u] for j in c_ids[v]):
        H.add_edge(u,v)

def weight_cluster_edge(G,H,u,v,c_ids):
    return sum(1 if G.edge( i,j ) else 0 for i in c_ids[u] for j in c_ids[v])
            



def get_cluster_graph(G,c_ids):
    H = gt.Graph(directed=False)
    n,m = len(c_ids), G.num_vertices()
    H.add_vertex(n)
    counts = np.zeros((n,n))
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
        if k == 0: continue
        top_k = set(np.argsort(d[v])[1:k+1])
        NE += (len(top_k.intersection(neighbors)) / len(top_k.union(neighbors)))
    return NE / G.num_vertices()


def compute_cluster_metric(G,X,c_ids):
    H = get_cluster_graph(G,c_ids)
    low_d_clusters = find_cluster_centers(X,c_ids)

    return get_neighborhood(H,low_d_clusters)

def get_cluster_distances(G,c_ids):
    n = len(c_ids)
    s = np.zeros((n,n))

    for u in range(n):
        for v in range(u+1):
            x = sum(1 if G.edge(i,j) else 0 for i in c_ids[u] for j in c_ids[v])
            x = np.exp(-(x**2))
            s[u,v] = x
            s[v,u] = s[u,v]
    d = 1-(s/np.max(s,axis=0))
    return (d + d.T) / 2

from metrics import get_stress, chen_neighborhood,cluster_preservation2
def compute_graph_cluster_metrics(G,X,c_ids):
    # c_ids,state = get_cluster_ids(G)
    # H = get_cluster_graph(G,c_ids)
    # high_d = apsp(H)
    high_d = get_cluster_distances(G,c_ids)
    low_d = find_cluster_centers(X,c_ids)

    return get_stress(low_d,high_d),cluster_preservation2(high_d,pairwise_distances(low_d),c_ids)


if __name__ == "__main__":
    G = gt.load_graph("graphs/connected_watts_400.dot")


    from modules.L2G import L2G,get_w
    w = get_w(G,k=30)
    X = L2G(apsp(G),weighted=True,w=w).solve(100)
    # X = L2G(apsp(G),weighted=False).solve(100)

    m1,m2,c_ids,state = compute_graph_cluster_metrics(G,X)
    print(m1,m2)

    def get_c_id(v):
        for i,c in enumerate(c_ids):
            if v in c:
                return i

    clusters = [cmap[get_c_id(v)] for v in G.iter_vertices()]
    clust_vp = G.new_vp("string",vals=clusters)

    pos = G.new_vp("vector<float>")
    pos.set_2d_array(X.T)
    state.draw(pos=pos)
    # gt.graph_draw(G,pos=pos,vertex_fill_color=clust_vp)

    cX = find_cluster_centers(X,c_ids)
    import pylab 
    pylab.scatter(cX[:,0],cX[:,1])
    pylab.show()
