import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt
import pickle

import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix

norm = lambda x: np.linalg.norm(x,ord=2)



with open('data/grid_regressionNP.pkl', 'rb') as myfile:
    data = pickle.load(myfile)

print(len(data['blocks']['NP']))
print(data['blocks']['NP'].keys())
x = np.arange(len(data['blocks']['NP']))

#Use .format

index = lambda n,a,k: 'V{}_a{}_k{}'.format(n,a,k)

V = list(range(10,400,10))
A = list(range(1,30))[:10]
K = [2,4,6,8,12,18,24] + [i for i in range(50,401,50)]

metric = data['blocks']['NP']
Y = []

for v in [130]:
    for a in [5]:
        for k in K:
            Y.append(metric[index(v,a,k)])

plt.plot(np.arange(len(Y)),Y)
plt.show()
