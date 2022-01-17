import numpy as np
import graph_tool.all as gt
import scipy.io
import time
import modules.distance_matrix as distance_matrix
from SGD_MDS import SGD_MDS
from SGD_MDS2 import SGD_MDS2
import modules.graph_io as graph_io


#G = graph_io.load_graph("graphs/dwt_419.vna")
G = gt.lattice([10,10])
d = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

def layout(d):
    Y = SGD_MDS(d,weighted=False)
    Y.solve()



def timing(f, n, a):
    print(f.__name__)
    r = range(n)
    t1 = time.perf_counter()
    for i in r:
        f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a)
    t2 = time.perf_counter()
    print((t2-t1)/10)

timing(layout,1,d)

Y = SGD_MDS(d,weighted=False)
Y.solve()
pos = G.new_vp('vector<float>')
pos.set_2d_array(Y.X.T)

gt.graph_draw(G,pos=pos)
