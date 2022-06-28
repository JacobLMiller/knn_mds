#!/usr/bin/env python3

if __name__ == '__main__':
    #Driver script adapted from https://github.com/HanKruiger/tsNET
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Read a graph, and produce a layout with tsNET(*).')

    # Input
    parser.add_argument('input_graph')
    parser.add_argument('--opt_scale', '-s', type=int, default=0, help='Whether to optimize scaling parameter.')
    parser.add_argument('--k', '-k', type=int, default=25, help='Number of most-connected neighbors to find')
    parser.add_argument('--alpha', '-a', type=float, default=0.1, help='Strength of the repulsive force.')
    parser.add_argument('--epsilon', '-e', type=float, default=1e-7, help='Threshold for convergence.')
    parser.add_argument('--max_iter', '-m', type=int, default=500, help='Maximum number of iterations.')
    parser.add_argument('--learning_rate', '-l', type=str, default="convergent", help='Learning rate (hyper)parameter for optimization.')
    parser.add_argument('--output', '-o', type=str, help='Save layout to the specified file.')

    args = parser.parse_args()

    #Import needed libraries
    import os
    import time
    import graph_tool.all as gt
    import numpy as np

    #Import modules
    from modules.L2G import L2G, get_w
    def apsp(G,weights=None):
        d = np.array( [v for v in gt.shortest_distance(G,weights=weights)] ,dtype=float)
        return d

    #Check for valid input
    assert(os.path.isfile(args.input_graph))
    graph_name = os.path.splitext(os.path.basename(args.input_graph))[0]

    #Global hyperparameters
    max_iter = args.max_iter
    eps = args.epsilon
    opt_scale = args.opt_scale
    lr = args.learning_rate
    k = args.k
    alpha = args.alpha

    #Load input graph
    print('Reading graph: {0}...'.format(args.input_graph), end=' ', flush=True)
    G = gt.load_graph(args.input_graph)
    print('Done.')

    print('Input graph: {0}, (|V|, |E|) = ({1}, {2})'.format(graph_name, G.num_vertices(), G.num_edges()))

    #Start timer
    start = time.perf_counter()

    #Get all-pairs-shortest-path matrix
    print('Computing SPDM...'.format(graph_name), end=' ', flush=True)
    d = apsp(G)
    print("Done.")

    #Get weight matrix
    w = get_w(G,k=k)

    #Perform optimization from SGD_MDS_sphere module
    X = L2G(d,weighted = True,w=w).solve(
        num_iter = args.max_iter,
        t=alpha,
        tol = args.epsilon
    )

    end = time.perf_counter()
    comp_time = end - start
    print('SMDS took {0:.2f} s.'.format(comp_time))

    print("-----------------------")

    #Save vectors to graph
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)

    #Save or display layout
    if args.output: gt.graph_draw(G,pos=pos, output=args.output)
    else: gt.graph_draw(G,pos=pos)
