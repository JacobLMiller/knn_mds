from SGD_MDS2 import SGD_MDS2
from SGD_MDS import SGD_MDS
from MDS_classic import MDS
import modules.distance_matrix as distance_matrix
import modules.layout_io as layout_io

import matplotlib.pyplot as plt
import numpy as np
import graph_tool.all as gt
import networkx as nx
import scipy.io

from metrics import get_neighborhood, get_norm_stress
from sklearn.metrics import pairwise_distances
import random


def display_stats(graph):
    G = gt.load_graph('graphs/{}.dot'.format(graph))
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)

    print(graph)
    print("num_vertices", G.num_vertices())
    print("num_edges",G.num_edges())
    print("diameter", np.max(d))
    print("global clustering coefficient", gt.global_clustering(G))
    print("average local clustering coefficient", sum(gt.local_clustering(G))/G.num_vertices())
    print("average degree", gt.vertex_average(G,'total'))
    print("edges/nodes", G.num_edges()/G.num_vertices())

    print()

def lin_log_random_graph(n=400):
    G = gt.Graph(directed=False)
    G.add_vertex(n=n)

    part  = n // 8
    half = n // 2

    for i in range(n):
        for j in range(i):
            c1, c2 = i // part, j // part


            if i == j and i < half and j < half:
                G.add_edge(i,j)
            elif i == j and random.random() < 0.5:
                G.add_edge(i,j)
            elif i < half and j < half and random.random() < 0.2:
                G.add_edge(i,j)
            elif i >= half and j >= half and random.random() < 0.05:
                G.add_edge(i,j)
            elif i < half and j >= half and random.random() < 0.1:
                G.add_edge(i,j)
    return G


def convert_graph(H):
    H = nx.convert_node_labels_to_integers(H)
    G = gt.Graph(directed=False)
    G.add_vertex(n=len(H.nodes()))
    for e in H.edges():
        G.add_edge(e[0],e[1])
    return G


import re
from io import StringIO

import pandas as pd
import graph_tool.stats as gts


def pajTOgt(filepath, directed = False, removeloops = True):
  if directed:
    g = gt.Graph(directed=True)
  else:
    g = gt.Graph(directed=False)

  #define edge and vertex properties
  g.edge_properties["weight"] = g.new_edge_property("double")
  g.vertex_properties["id"] = g.new_vertex_property("string")

  with open(filepath, encoding = "utf-8") as input_data:
    #create vertices
    for line in input_data:
        g.add_vertex(int(line.replace("*Vertices ", "").strip())) #add vertices
        break

    #label vertices
    for line in input_data: #keeps going for node labels
      if not line.strip() == '*Edges' or line.strip() == '*Arcs':
        v_id = int(line.split()[0]) - 1
        g.vertex_properties["id"][g.vertex(v_id)] = "".join(line.split()[1:])
      else:
        break

    #create weighted edges
    for line in input_data: #keeps going for edges
      linesplit = line.split()
      linesplit = [int(x) for x in linesplit[:2]] + [float(linesplit[2])]
      if linesplit[2] > 0:
        n1 = g.vertex(linesplit[0]-1)
        n2 = g.vertex(linesplit[1]-1)
        e = g.add_edge(n1, n2)
        g.edge_properties["weight"][e] = linesplit[2]

    if removeloops:
      gts.remove_self_loops(g)

    return g

import autograd.numpy as np  
from autograd import grad


d = np.array( [
                [0,1,np.sqrt(2)],
                [1,0,1],
                [np.sqrt(2),1,0]
            ] )
def tanh(x):                 # Define a function
    stress = 0
    for i in range(len(x)):
        for j in range(i):
            stress += pow(np.linalg.norm(x[i]-x[j]) - d[i][j],2)
    return stress
grad_tanh = grad(tanh)
print(grad_tanh(np.array([
                        [0,2],
                        [1,1],
                        [1,0]
                        ])))
