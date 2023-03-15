from modules import thesne
from modules import layout_io
from modules import distance_matrix

import graph_tool.all as gt 
import numpy as np

from sklearn.metrics import pairwise_distances

def get_neighborhood(X,d,rg = 2):
    """
    How well do the local neighborhoods represent the theoretical neighborhoods?
    Closer to 1 is better.
    Measure of percision: ratio of true positives to true positives+false positives
    """
    norm = np.linalg.norm
    def get_k_embedded(X,k_t):
        dist_mat = pairwise_distances(X)
        return [np.argpartition(dist_mat[i],len(k_t[i]))[:len(k_t[i])] for i in range(len(dist_mat))]

    k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0] for i in range(len(d))]

    k_embedded = get_k_embedded(X,k_theory)

    sum = 0
    for i in range(len(X)):
        count_intersect = 0
        for j in range(len(k_theory[i])):
            if k_theory[i][j] in k_embedded[i]:
                count_intersect += 1
        sum += count_intersect/ len(k_theory[i])

    return sum/len(X)

def get_stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(len(X)):
            stress += pow(d[i][j] - np.linalg.norm(X[i]-X[j]),2)
    return stress / np.sum(np.square(d))

def get_tsnet_layout(d,graph_name,p=30,interpolate=False,a=0.9,verbose=False):
    n = 2000
    momentum = 0.5
    tolerance = 1e-7
    window_size = 40

    # Cost function parameters
    r_eps = 0.05

    # Phase 2 cost function parameters
    lambdas_2 = [1, 1.2, 0]

    # Phase 3 cost function parameters
    lambdas_3 = [1, 0.01, 0.6]

    # Read input graph

    print('Input graph: {0},'.format(graph_name))

    # Load the PivotMDS layout for initial placement
    Y_init = None

    # Time the method including SPDM calculations

    # Compute the shortest-path distance matrix.
    X = d

    # sigma = 600 if graph_name == 'EVA.dot' else 100

    # The actual optimization is done in the thesne module.
    Y,hist = thesne.tsnet(
        X, output_dims=2, random_state=1, perplexity=p, n_epochs=n,
        Y=Y_init,
        initial_lr=50, final_lr=50, lr_switch=n // 2,
        initial_momentum=momentum, final_momentum=momentum, momentum_switch=n // 2,
        initial_l_kl=lambdas_2[0], final_l_kl=lambdas_3[0], l_kl_switch=n // 2,
        initial_l_c=lambdas_2[1], final_l_c=lambdas_3[1], l_c_switch=n // 2,
        initial_l_r=lambdas_2[2], final_l_r=lambdas_3[2], l_r_switch=n // 2,
        r_eps=r_eps, autostop=tolerance, window_size=window_size,
        verbose=verbose, a = a, interpolate=interpolate
    )

    Y = layout_io.normalize_layout(Y)

    return Y

def tsnet_exp():
    Gs = [
        gt.load_graph("graphs/block_400.dot"),
        gt.load_graph("graphs/connected_watts_300.dot"),
        gt.load_graph("graphs/12square.dot")
    ]
    names = ['blocks','watts','grid']

    for i,G in enumerate(Gs):
        d = distance_matrix.get_distance_matrix(G,'spdm')
        d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)

        scores = list() 

        for p in range(10,101,5):
            print(f"Graph: {names[i]}, perplexity: {p}")
            NP, stress = 0,0
            for _ in range(15):
                Y = get_tsnet_layout(d,"block_400",p)
                NP += get_neighborhood(Y,d_norm,1)
                stress += get_stress(Y,d)
            scores.append([NP/15, stress/15])

            # Convert layout to vertex property
        # pos = G.new_vp('vector<float>')
        # pos.set_2d_array(Y.T)

        # # Show layout on the screen
        # gt.graph_draw(G, pos=pos)

        np.savetxt(f"tsnet-scores-{names[i]}.txt", np.array(scores))
        print(get_neighborhood(Y,d,1))    


def penguins(k,weight=True):
    import pandas as pd
    data = pd.read_csv("palmerpenguins.csv")


    labels, index_map = pd.factorize(data['species'])


    gender_map = {'male': 0, 'female': 1}
    marker_map = {0: "^", 1: "o"}
    markers,ind_map = pd.factorize(data["sex"])
    male = np.where(markers == 0)
    female = np.where(markers == 1)

    cmap = {0: "red", 
            1: "blue",
            2: "orange",
            3: "tab:red",
            4: "tab:purple"}

    C = np.array([cmap[c] for c in labels])    

    print(data.head())
    Y = data.drop(['rowid', 'species', 'island', 'year'],axis=1).to_numpy()
    Y[male,4] = 1
    Y[female,4] = 0
    Y = Y.astype(np.float64)


    Y /= np.max(Y,axis=0)


    from sklearn.metrics import pairwise_distances 
    d = pairwise_distances(Y)

    G = gt.Graph(directed=False)
    G.add_vertex(d.shape[0])

    kp = 5
    for i, row in enumerate(d):
        inds = np.argsort(row)
        for u in range(1,kp+1):
            G.add_edge(i,u)

    from modules import L2G

    w = L2G.get_w(G,k=k)
    X = L2G.L2G(d,weighted=False).solve(100)

    import pylab 

    pylab.scatter(X[:,0],X[:,1], 20, C)
    pylab.savefig(f"drawings/penguins_l2g_k={k}.png")
    pylab.clf()

if __name__ == "__main__":
    tsnet_exp()
    # for k in [2,5,10,25,50,100,200]:
    #     penguins(k)
    # penguins(300,False)
    # G = gt.load_graph("graphs/block_model_400.dot")

    # d = distance_matrix.get_distance_matrix(G,'spdm')

    # for a in np.linspace(0,1,10):
    #     X = get_tsnet_layout(d,'watts',interpolate=True,a=a)
        
    #     pos = G.new_vp('vector<float>')
    #     pos.set_2d_array(X.T)

    #     gt.graph_draw(G,pos,output=f"drawings/interpolate_test_{a}.pdf")        
