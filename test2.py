import numpy as np
import graph_tool.all as gt
import scipy.io


A = scipy.io.mmread('graphs/oscil_dcop_01.mtx').toarray()
G = gt.Graph(directed=False)
G.add_edge_list(np.transpose(A.nonzero()))
gt.remove_parallel_edges(G)
gt.remove_self_loops(G)

remove = []
for v in G.iter_vertices():
    if len(list(G.iter_all_neighbors(v))) == 0:
        remove.append(v)

G.remove_vertex(remove)        

G.save('graphs/oscil.dot')
