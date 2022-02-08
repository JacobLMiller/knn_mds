import numpy as np
import graph_tool.all as gt
import modules.layout_io as layout_io
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne

from SGD_MDS2 import SGD_MDS2

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

def get_tsnet_layout(d,graph_name):
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

    sigma = 600 if graph_name == 'EVA.dot' else 100

    # The actual optimization is done in the thesne module.
    Y = thesne.tsnet(
        X, output_dims=2, random_state=1, perplexity=sigma, n_epochs=n,
        Y=Y_init,
        initial_lr=50, final_lr=50, lr_switch=n // 2,
        initial_momentum=momentum, final_momentum=momentum, momentum_switch=n // 2,
        initial_l_kl=lambdas_2[0], final_l_kl=lambdas_3[0], l_kl_switch=n // 2,
        initial_l_c=lambdas_2[1], final_l_c=lambdas_3[1], l_c_switch=n // 2,
        initial_l_r=lambdas_2[2], final_l_r=lambdas_3[2], l_r_switch=n // 2,
        r_eps=r_eps, autostop=tolerance, window_size=window_size,
        verbose=True
    )

    Y = layout_io.normalize_layout(Y)



    # Convert layout to vertex property
    # pos = g.new_vp('vector<float>')
    # pos.set_2d_array(Y.T)

    # Show layout on the screen
    #gt.graph_draw(g, pos=pos)
    return Y

def get_SGD_layout(d):
    return SGD_MDS2(d,weighted=False).solve()

def get_LG_layout(d,k,G):
    A = gt.adjacency(G).toarray()
    A = np.linalg.matrix_power(A,5)
    A += np.random.normal(scale=0.01,size=A.shape)

    #k = 399
    k_nearest = [np.argpartition(A[i],-k)[-k:] for i in range(len(A))]

    n = G.num_vertices()
    N = 0
    w = np.asarray([[ 1e-5 if i != j else 0 for i in range(len(A))] for j in range(len(A))])
    for i in range(len(A)):
        for j in k_nearest[i]:
            if i != j:
                w[i][j] = 1
                w[j][i] = 1

    for i in range(len(w)):
        for j in range(i):
            if w[i][j] == 1:
                N += 1

    Nc = (n*(n-1))/2 - N
    t = (N/Nc)*np.median(d)*0.1

    return SGD_MDS2(d,weighted=True,w=w).solve(num_iter=15,t=t,debug=False)

def calc_tsnet(d,d_norm,graph):
    X = get_tsnet_layout(d_norm,graph)
    X_norm = layout_io.normalize_layout(X)

    return get_neighborhood(X_norm,d),get_stress(X_norm,d_norm)

def calc_SGD(d,d_norm):
    X = get_SGD_layout(d_norm)
    X_norm = layout_io.normalize_layout(X)

    return get_neighborhood(X_norm,d),get_stress(X_norm,d_norm)

def calc_LG_low(d,d_norm,G):
    X = get_LG_layout(d_norm,8,G)
    X_norm = layout_io.normalize_layout(X)

    return get_neighborhood(X_norm,d),get_stress(X_norm,d_norm)

def calc_LG_high(d,d_norm,G):
    X = get_LG_layout(d_norm,int(0.8*G.num_vertices()),G)
    X_norm = layout_io.normalize_layout(X)

    return get_neighborhood(X_norm,d),get_stress(X_norm,d_norm)



def main(n=5):
    import os
    import pickle
    import copy

    path = 'tsnet-graphs/'
    graph_paths = os.listdir(path)

    template_score = {
        'NP' : [None for i in range(n)],
        'stress': [None for i in range(n)]
    }

    template_alg = {
        'tsnet': copy.deepcopy(template_score),
        'SGD': copy.deepcopy(template_score),
        'LG_low': copy.deepcopy(template_score),
        'LG_high': copy.deepcopy(template_score)
    }

    scores = {key : copy.deepcopy(template_alg) for key in graph_paths}

    for graph in graph_paths:
        G = gt.load_graph(path+graph)
        d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
        d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

        print("Graph: " + graph)
        print("-----------------------------------------------------------")

        for i in range(n):
            print("Iteration number ", i)
            NP,stress = calc_tsnet(d,d_norm,graph)
            scores[graph]['tsnet']['NP'][i] = NP
            scores[graph]['tsnet']['stress'][i] = stress

            NP,stress = calc_SGD(d,d_norm)
            scores[graph]['SGD']['NP'][i] = NP
            scores[graph]['SGD']['stress'][i] = stress

            NP,stress = calc_LG_low(d,d_norm,G)
            scores[graph]['LG_low']['NP'][i] = NP
            scores[graph]['LG_low']['stress'][i] = stress

            NP,stress = calc_LG_high(d,d_norm,G)
            scores[graph]['LG_high']['NP'][i] = NP
            scores[graph]['LG_high']['stress'][i] = stress


        with open('data/tsnet-repeat.pkl','wb') as myfile:
            pickle.dump(scores,myfile)
        myfile.close()


if __name__ == "__main__":
    main()
