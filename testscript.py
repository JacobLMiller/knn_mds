import pickle
import numpy as np
import networkx as nx
import graph_tool.all as gt
from SGD_MDS import SGD_MDS

import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix

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

with open('experiment1.pkl', 'rb') as myfile:
   data = pickle.load(myfile)
graphs = ['dwt_1005.vna','dwt_419.vna','small_block.dot','bigger_block.dot','jazz.vna','block_2000.vna']


G = 'dwt_419.vna'
stress = {}
neighbors = {}
for count in range(5):
    g = G+str(count)
    H = graph_io.load_graph('graphs/'+ G)
    # X = layout_io.normalize_layout(data[g]['tsnet']['layout'])
    #
    # # Convert layout to vertex property
    # pos = H.new_vp('vector<float>')
    # pos.set_2d_array(X.T)

    #gt.graph_draw(H,pos=pos,output='drawings/tsnet'+str(count)+'.png')
    for i in data[g]['weight']:
        stress[i] = []
        neighbors[i] = []
        stress[i].append(data[g]['weight'][i]['stress'])
        #neighbors[i].append(data[g]['weight'][i]['neighbor'])
        X = data[g]['weight'][i]['layout']
        neighbors[i].append(get_neighborhood(X,distance_matrix.get_distance_matrix(H,'spdm',normalize=False,verbose=False),rg=2))

        # H = graph_io.load_graph('graphs/'+ G)
        # X = layout_io.normalize_layout(data[g]['weight'][i]['layout'])
        #
        #
        # # Convert layout to vertex property
        # pos = H.new_vp('vector<float>')
        # pos.set_2d_array(X.T)
        #
        # gt.graph_draw(H,pos=pos,output='drawings/weighted-k'+str(i)+'.png')
            #print(data['small_block.dot0']['weight'][i][j])

stress_median = {}
neighbor_median = {}
for i in stress:
    stress[i] = np.array(stress[i]).mean()
    neighbors[i] = 1-np.array(neighbors[i]).mean()
    stress_median[i] = np.median(np.array(stress[i]))
    neighbor_median[i] = 1-np.median(np.array(neighbors[i]))
print(stress)
print(neighbors)
print(neighbor_median)

import matplotlib.pyplot as plt

plt.suptitle(G + " |V|: " + str(data[G+str(count)]['attributes']['|V|']) + " |E|: " + str(data[G+str(count)]['attributes']['|E|']) + " \ndiameter: " + str(data[G+str(count)]['attributes']['diameter']) + " Cluster coefficient: " + str(data[G+str(count)]['attributes']['clustering-coefficient']))

L = np.array(list(stress.keys()))
plt.xticks(ticks=np.arange(len(stress)),labels=L.astype('str'))
plt.plot(np.arange(len(stress)),list(stress_median.values()),'o-', label="median distortion")
plt.plot(np.arange(len(neighbors)),list(neighbor_median.values()),'o-', label="median 1-neighbor_score")
plt.legend()
plt.savefig('figures/dwt_419.png')
