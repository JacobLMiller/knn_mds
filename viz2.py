import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt
import pickle

import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix

norm = lambda x: np.linalg.norm(x,ord=2)

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

def dist(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += abs(np.linalg.norm(X[i]-X[j])-d[i][j])/d[i][j]
    return stress/choose(len(X),2)

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
            k = len(k_theory[i])
            if k_theory[i][j] in k_embedded[i]:
                count_intersect += 1
        sum += count_intersect/(2*k - count_intersect)

    return sum/len(X)

def stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow(np.linalg.norm(X[i]-X[j])-d[i][j],2)
    return pow(stress,0.5)

with open('push_data/lg_random_graphs1.pkl', 'rb') as myfile:
    data = pickle.load(myfile)

with open('data/sgd_random_graphs.pkl','rb') as myfile:
    sgd_data = pickle.load(myfile)

with open('data/tsnet_random_graphs.pkl','rb') as myfile:
    tsnet_data = pickle.load(myfile)

print(data.keys())



metric = 'NP'






# plt.plot([1 for _ in range(5)],LG_low,'o',label="LG_low")
# plt.plot([2 for _ in range(5)],LG_high,'o',label="LG_high")
#
#
# plt.legend()
# plt.show()
# plt.clf()

row_labels = list(data.keys())
row_labels.sort()


max_graph = {graph:0 for graph in row_labels}

column_labels = ["LG,k=18","LG,k= |V|", 'SGD', 'tsNET']

cell_data = []
count_tsnet, count_sgd = 0,0
for row in row_labels:
    if data[row][metric][0] > 0:
        lg_low = data[row][metric][0]
        lg_high = data[row][metric][-1]
        if row in sgd_data:
            sgd = sgd_data[row][metric]
        else:
            sgd = 0
        if row in tsnet_data:
            tsnet = tsnet_data[row][metric]
        else:
            tsnet = 0
        this_row = np.array([round(lg_low,5),round(lg_high,5),round(sgd,5),round(tsnet,5)])
        cell_data.append(this_row)
        max_graph[row] = np.argmin(this_row)
    else:
        cell_data.append([0,0,0,0])

print("low performed better on ", count_tsnet)
print("hgih performed better on ", count_sgd)

plt.suptitle("NP")
table = plt.table(cellText=cell_data,
                      rowLabels=row_labels,
                      colLabels=column_labels,
                      loc='center',
                      colWidths=[0.2,0.2,0.2,0.2],
                      cellLoc='center')

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

for (row, col), cell in table.get_celld().items():
    print(row)
    print(row_labels[row-1])
    print()
    if row == 0: continue
    if (max_graph[row_labels[row-1]] == col):
        cell.set_text_props(fontproperties=FontProperties(weight='bold'))

plt.xticks([])
plt.yticks([])
plt.axis('off')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

#plt.show()
#plt.savefig('prelim_table_stress.eps')

plt.clf()

graphs = list(data.keys())
consider = graphs[6]
X = data[consider]['NP']
Y = data[consider]['stress']
print(consider)
print(X)

plt.plot(X,Y)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
