import pickle
import numpy as np
import networkx as nx
import graph_tool.all as gt


with open('test_gather.pkl', 'rb') as myfile:
    myObj = pickle.load(myfile)

print(myObj)
