import pickle
import numpy as np
import networkx as nx
import graph_tool.all as gt
from SGD_MDS import SGD_MDS

import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix


with open('experiment1.pkl', 'rb') as myfile:
   data = pickle.load(myfile)
graphs = ['dwt_1005.vna','dwt_419.vna','small_block.dot','bigger_block.dot','jazz.vna','block_2000.vna']


G = 'block_2000.vna'
stress = {}
neighbors = {}
for count in range(5):
    g = G+str(count)
    for i in data[g]['weight']:
        stress[i] = []
        neighbors[i] = []
        for j in data[g]['weight'][i]:
            stress[i].append(data[g]['weight'][i]['stress'])
            neighbors[i].append(data[g]['weight'][i]['neighbor'])
            #print(data['small_block.dot0']['weight'][i][j])
for i in stress:
    stress[i] = np.array(stress[i]).mean()
    neighbors[i] = np.array(neighbors[i]).mean()
print(stress)
print(neighbors)

import matplotlib.pyplot as plt
#plt.xticks(ticks=np.arange(len(stress)),labels=np.asarray(stress.keys()).astype('str'))
plt.plot(np.arange(len(stress)),list(stress.values()),'o-', label="stress")
plt.plot(np.arange(len(stress)),list(neighbors.values()),'o-', label="neighbor")
plt.legend()
plt.show()
