import numpy as np 
import pylab
from sklearn.metrics import pairwise_distances
from metrics import get_stress, chen_neighborhood, cluster_distance, cluster_preservation, cluster_preservation2, mahalonobis_metric


# %%
def print_metrics(X,d,H,c_ids):
    stress = get_stress(X,d)
    NH = 1-chen_neighborhood(d,X,k=7)
    CD = cluster_distance(H,X,c_ids)
    print(f"Stress is {stress}; NH is {NH}, CD is {CD}")
    return stress, NH, CD

c1 = list()
for i in range(5):
    for j in range(5):
        for k in range(5):
            c1.append([i,j,k])

grid = np.array(c1)
c1 = grid + np.array([5,0,0])
c1.shape

# %%
gen_cluster = lambda s: np.random.normal(scale=s, size=[100,3])

c2 = gen_cluster(1) + np.array([0,10,-10])
c3 = (grid*0.25) + np.array([0,5,-5])
c4 = gen_cluster(1)
c5 = gen_cluster(0.25) + np.array([5,3,2]) 



# %%
clusters = [c1,c2,c3,c4,c5]

# %%
data = np.concatenate(clusters,axis=0)
data.shape

def label_clusters(sizes):
    return sum([[i] * size for i,size in enumerate(sizes)], [])
sizes = [c.shape[0] for c in clusters]
C = label_clusters(sizes)
data_name = "3d"
c_ids = C




d = pairwise_distances(data)
tmp = [[] for _ in np.unique(c_ids)]
[tmp[c].append(i) for i,c in enumerate(c_ids)]
c_ids = tmp

data = list()
for _ in range(1000):
    vec = np.random.normal(0,1,size=(3))
    vec /= np.linalg.norm(vec)
    data.append(vec)
    
data = np.array(data)

fig = pylab.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(data[:,0],data[:,1],data[:,2])
# print_metrics(data,d,data,c_ids)
# fig.savefig("figures/3d-og.png")
pylab.show()
pylab.clf()


d = pairwise_distances(data)
from modules.L2G import L2G
X = L2G(d,weighted=False).solve(100,rep=False)
pylab.scatter(X[:,0],X[:,1],alpha=0.5)
# s,n,c = print_metrics(X,d,data,c_ids)
# pylab.suptitle(f"mds\n stress: {s}\nNH: {n}")
pylab.show()
# pylab.savefig(f"figures/{data_name}-mds.png")
# m_scores = (s,n,c)
