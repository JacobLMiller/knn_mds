



def experiment(n=5):
    import os
    import pickle
    import copy

    path = 'example_graphs/'
    graph_paths = os.listdir(path)

    graph_paths = list( map(lambda s: s.split('.')[0], graph_paths) )
    #graph_paths = ['custom_cluster_100']
    print(graph_paths)

    adjacency_len = len( np.linspace(5,100,8) )

    zeros = lambda s: np.zeros(s)
    alg_dict = {'NP': zeros(adjacency_len), 'stress': zeros(adjacency_len), 'cost': zeros(adjacency_len)}
    exp = {'iterations': copy.deepcopy(alg_dict),
           'matrix_power': copy.deepcopy(alg_dict),
           'alpha': copy.deepcopy(alg_dict)}


    graph_dict = {key: copy.deepcopy(exp) for key in graph_paths}

    for graph in graph_paths:
        print(graph)
        G = gt.load_graph(path+graph + '.dot')
        d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
        d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

        CC = G.num_edges() // G.num_vertices()
        a = 3 if CC < 4 else 4 if CC < 8 else 5

        print("Graph: " + graph)
        print("-----------------------------------------------------------")


        iteration_exp(graph,G,d,d_norm,a)
        matrix_exp(graph,G,d,d_norm)
        alpha_exp(graph,G,d,d_norm,a)
