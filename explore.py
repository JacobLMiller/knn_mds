from SGD_MDS import SGD_MDS, k_nearest_embedded
import graph_tool.all as gt
import time
import numpy as np

import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne


"""
For normal GD-SGD, the distance matrix is symmetric, because we assume an unweighted graph. Could we improve the clustering
by imposing a directed graph so that each node has EXACTLY k out-degree... Maybe need to modify the algorithm to only update 1 node at a time, rather than 2.

"""

G = graph_io.load_graph('graphs/block_2000.vna')
#G = gt.lattice([10,10])
print(G)

K = [2,4,16,32]

d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)

Y = SGD_MDS(d,weighted=True,k=2)
#print(Y.w)
Y.solve(30)

adj = Y.w.copy()
H = gt.Graph(directed=False)
H.add_vertex(len(adj))
H.add_edge_list(np.argwhere(adj >= 0.5))
gt.remove_parallel_edges(H)
print('hello')
print(adj)
print(H)

pos = H.new_vp('vector<float>')
pos.set_2d_array(Y.X.T)

gt.graph_draw(H,pos=pos)

pos = G.new_vp('vector<float>')
pos.set_2d_array(Y.X.T)

gt.graph_draw(G,pos=pos)
