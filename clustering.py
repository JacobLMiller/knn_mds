import graph_tool.all as gt
import networkx as nx


def convert_graph(H):
    H = nx.convert_node_labels_to_integers(H)
    G = gt.Graph(directed=False)
    G.add_vertex(n=len(H.nodes()))
    for e in H.edges():
        G.add_edge(e[0],e[1])
    return G


G = convert_graph(nx.erdos_renyi_graph(100, 0.5,directed=False))
print(gt.global_clustering(G))
