import graph_tool.all as gt

import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne
import time
import numpy as np

from SGD_MDS import SGD_MDS, k_nearest_embedded

norm = lambda x: np.linalg.norm(x,ord=2)


#G = gt.lattice([10,10])
#G = graph_io.load_graph('small_block.dot')
#d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)


def get_tsnet_layout(G,d,graph_name):
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
    g = G

    print('Input graph: {0}, (|V|, |E|) = ({1}, {2})'.format(graph_name, g.num_vertices(), g.num_edges()))

    # Load the PivotMDS layout for initial placement
    Y_init = None

    # Time the method including SPDM calculations
    start_time = time.time()

    # Compute the shortest-path distance matrix.
    X = d

    sigma = 100 if graph_name == 'jazz.vna' or graph_name == 'bigger_block.dot' else 40

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

    end_time = time.time()
    comp_time = end_time - start_time
    print('tsNET took {0:.2f} s.'.format(comp_time))

    # Convert layout to vertex property
    pos = g.new_vp('vector<float>')
    pos.set_2d_array(Y.T)

    # Show layout on the screen
    #gt.graph_draw(g, pos=pos)
    return Y

def get_unweighted(G,d):
    print("Beginning Unweighted")
    Y = SGD_MDS(d,weighted=False)
    Y.solve(30)
    X = layout_io.normalize_layout(Y.X)

    # Convert layout to vertex property
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    return Y.X

def get_weighted(G,d):
    K = [1,2,4,8,16,32,64,128]
    weights = {}
    print("Beginning weighted tests")
    for k in K:
        print('k = '.format(k), end='\r')
        print()
        m = k if k < G.num_vertices() else G.num_vertices()-1

        print(m)
        #print(m)

        Y = SGD_MDS(d,weighted=True,k=m)
        Y.solve(30)
        X = layout_io.normalize_layout(Y.X)

        # Convert layout to vertex property
        pos = G.new_vp('vector<float>')
        pos.set_2d_array(X.T)

        gt.graph_draw(G,pos=pos,output='drawings/weighted-k' + str(k) + ".png")

        weights[k] = {'layout': Y.X,
                      'stress': get_stress(Y.X,d),
                      'neighbor': get_neighborhood(Y.X,d)}

    Y = SGD_MDS(d,weighted=True,k=G.num_vertices()-1)
    Y.solve(30)
    X = layout_io.normalize_layout(Y.X)

    # Convert layout to vertex property
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)

    gt.graph_draw(G,pos=pos,output='drawings/weighted-k' + str(k) + ".png")

    weights['n'] = {'layout': Y.X,
                  'stress': get_stress(Y.X,d),
                  'neighbor': get_neighborhood(Y.X,d)}
    return weights

def get_stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow((norm(X[i]-X[j])-d[i][j])/d[i][j],2)
    return pow(stress,0.5)

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

#graphs = ['lesmis.vna', 'dwt_1005.vna','jazz.vna','bigger_block.dot','small_block.dot','grid17.vna','visbrazil.vna']
graphs = ['lesmis.vna']
layouts = {}

for g in graphs:
    print("------------------------------")
    print("------------------------------")
    print("------------------------------")
    print("Graph: " + g)

    G = graph_io.load_graph("graphs/" + g)
    d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
    for j in range(5):
        name = g + str(j)
        #tsnet = get_tsnet_layout(G,d,g)
        print("-----------------------")
        #unweight = get_unweighted(G,d)
        print("-----------------------")
        weight = get_weighted(G,d)

        #D = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)


        layouts[name] = {'tsnet': {'layout': tsnet},
                      'unweight': {'layout':unweight},
                      'weight': weight,

                      'attributes': {
                        'name': g,
                        '|V|': G.num_vertices(),
                        '|E|': G.num_edges(),
                        'diameter': np.max(d),
                        'clustering-coefficient': gt.global_clustering(G)
                      }}

        for i in layouts[name].keys():
            if i != 'attributes' and i != 'weight':
                layouts[name][i]['stress'] = get_stress(layouts[name][i]['layout'],d)
                layouts[name][i]['neighbor'] = get_neighborhood(layouts[name][i]['layout'],d)

import pickle
with open('experiment.pkl', 'wb') as myfile:
    pickle.dump(layouts, myfile)
