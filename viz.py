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
            if k_theory[i][j] in k_embedded[i]:
                count_intersect += 1
        sum += count_intersect/(len(k_theory[i])+len(k_embedded[i])-count_intersect)

    return sum/len(X)

def stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow(np.linalg.norm(X[i]-X[j])-d[i][j],2)
    return pow(stress,0.5)

with open('A_experiments.pkl', 'rb') as myfile:
    final = pickle.load(myfile)

K = [1,2,3,4,5,6,7]

key = list(final.keys())[0]
print(key)

data = []
neighbors = []
G = graph_io.load_graph("graphs/" + key)
d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)
for i in final[key]:
    first = list(final[key][i].keys())[0]
    Xs = [layout_io.normalize_layout(final[key][i][first][j]) for j in range(5)]
    dists = [dist(x,d_norm) for x in Xs]
    neighbor = [1-get_neighborhood(x,d) for x in Xs]
    data.append(dists)
    neighbors.append(neighbor)

    pos = G.new_vp('vector<float>')
    pos.set_2d_array(Xs[0].T)

    gt.graph_draw(G,pos=pos,output='drawings/update/dwt419_A'+str(i)+'k' + str(first) +'.png')

print(dists)
print(len(data))
data = np.array(data)
neighbors = np.array(neighbors)

x = np.arange(len(K))
y = [np.median(data[i]) for i in range(len(x))]
print(len(y) == len(x))

L = np.array(K)


#
plt.suptitle('Scores on |V| = 419 dwt_419 mesh (reject all pairs not in k set)')
plt.plot(x,y,label='Average Distortion')
plt.plot(x,[np.median(neighbors[i]) for i in range(len(x))],label='Average Neighborhoood Score')
plt.xticks(ticks=np.arange(len(y)),labels=L.astype('str'))
plt.legend()
plt.ylim(0,1)
plt.show()
