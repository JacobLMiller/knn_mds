import numpy as np
import networkx as nx
import graph_tool.all as gt

graphs = nx.read_graph6('ER_v=100.g6')
print(len(graphs))

H = graphs[15]

G = gt.Graph(directed=False)
G.add_vertex(n=H.number_of_nodes())
for e1,e2 in H.edges():
    G.add_edge(e1,e2)

print(G)
G.save('graphs/test_ER.dot')
