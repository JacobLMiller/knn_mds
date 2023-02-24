import numpy as np
import graph_tool.all as gt
import modules.layout_io as layout_io
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne

# from SGD_MDS2 import SGD
# from SGD_MDS_debug import SGD_d
# from SGD_MDS import SGD_MDS

from sklearn.metrics import pairwise_distances

from metrics import get_stress,get_neighborhood
from graph_metrics import compute_graph_cluster_metrics, get_cluster_ids

import matplotlib.pyplot as plt

import gc

import time

from tqdm import tqdm 

def diffusion_weights(d,a=5, k = 20, sigma=1):
    #Transform distance matrix
    diff = np.exp( -(d**2) / (sigma **2) )
    diff /= np.sum(diff,axis=0)

    #Sum powers from 1 to a
    mp = np.linalg.matrix_power
    A = sum( pow(0.1,i) * mp(diff,i) for i in range(1,a+1) )

    #Find k largest points for each row 
    Neighbors = set()
    for i in range(diff.shape[0]):
        args = np.argsort(A[i])[::-1][1:k+1]
        for j in args:
            Neighbors.add( (int(i),int(j)) )

    #Set pairs to 1
    w = np.zeros_like(diff)
    for i,j in Neighbors:
        w[i,j] = 1
        w[j,i] = 1
    return w


def draw(G,X,output=None):
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    #
    if output:
        gt.graph_draw(G,pos=pos,output=output)
    else:
        gt.graph_draw(G,pos=pos)

from modules import L2G
def embed_l2g(d,d_norm,num_params,name,G,rep=True,log=True,f_name="test"):
    K = [4,8,32,128,512]
    NE,stress,times = np.zeros_like(K,dtype=float), np.zeros_like(K,dtype=float), np.zeros_like(K,dtype=float)
    Xs = list()
    for i, k in enumerate(K):
        k = int(k) if k < G.num_vertices() else G.num_vertices() - 1

        start = time.perf_counter()
        w = L2G.get_w(G,k=k)
        Y = L2G.L2G(d,w=w,weighted=True)
        X = Y.solve(70,t=0.1,rep=rep,log=log)       
        end = time.perf_counter()

        # NE[i] = get_neighborhood(X,d)
        # stress[i] = get_stress(X,d_norm)
        # times[i] = end-start

        outstr = f"drawings/{f_name}/{name}_{k}.png"
        draw(G,X,outstr)
        Xs.append(X)

    return Xs

def embed_l2g_transformation(d,d_norm,num_params,name,G,f_name="transformation"):
    K = [4,8,32,128,512]
    NE,stress,times = np.zeros_like(K,dtype=float), np.zeros_like(K,float), np.zeros_like(K,float)
    Xs = list()
    outs = list()
    for i, k in enumerate(K):
        k = int(k) if k < G.num_vertices() else G.num_vertices() - 1

        start = time.perf_counter()
        w = diffusion_weights(d,20,k=k)
        Y = L2G.L2G(d,w=w,weighted=True)
        X = Y.solve(200,t=0.1,rep=False)       
        end = time.perf_counter()

        # NE[i] = get_neighborhood(X,d)
        # stress[i] = get_stress(X,d_norm)
        # times[i] = end-start

        outstr = f"drawings/{f_name}/{name}_{k}.png"
        # draw(G,X,outstr)
        outs.append(outstr)
        Xs.append(X)

    return Xs,outs

from resist import get_tsnet_layout
def interp(d,d_norm,num_params,name,G):

    alphas = np.linspace(0,1,num_params)
    NE,stress = np.zeros_like(alphas), np.zeros_like(alphas)
    outs = list()
    for i,a in enumerate(alphas):
        X = get_tsnet_layout(d,name, interpolate=True, a=a,verbose=False)
        NE[i] = get_neighborhood(X,d)
        stress[i] = get_stress(X,d_norm)

        outstr=f"drawings/interp/{name}_{a}.png"
        # draw(G,X,outstr)
        outs.append(outstr)

    return X,outs

from sklearn.manifold import TSNE
def embed_tsne(d,d_norm,num_params,name,G):
    start = time.perf_counter()
    X = TSNE(learning_rate="auto",metric="precomputed").fit_transform(d)
    end = time.perf_counter()

    outstr = f"drawings/tsne/{name}.png"
    # draw(G,X,outstr)

    return [X], [outstr]

def embed_mds(d,d_norm,num_params,name,G):
    start = time.perf_counter()
    Y = L2G.L2G(d,weighted=False)
    X = Y.solve(70)
    end = time.perf_counter()


    outstr = f"drawings/mds/{name}.png"
    # draw(G,X,outstr)    
    return [X], [outstr]
    

def l2g_rep_all_pairs(d,d_norm,num_params,name,G):
    return embed_l2g(d,d_norm,num_params,name,G,rep=True,f_name="l2g_rep_all_pairs")

def l2g_rep_all_pairs_nolog(d,d_norm,num_params,name,G):
    return embed_l2g(d,d_norm,num_params,name,G,rep=True,log=False,f_name="l2g_rep_all_pairs_nolog")

def l2g_rep_some_pairs(d,d_norm,num_params,name,G):
    return embed_l2g(d,d_norm,num_params,name,G,rep=False,f_name="l2g_rep_comp_pairs")

def l2g_rep_some_pairs_nolog(d,d_norm,num_params,name,G):
    return embed_l2g(d,d_norm,num_params,name,G,rep=False,log=False,f_name="l2g_rep_comp_pairs_nolog")



def calc_linear(graph,G,d,d_norm):
    NP,stress = [],[]
    A = np.linspace(0,1,8)
    for a in A:

        n = 1500
        momentum = 0.5
        tolerance = 1e-7
        window_size = 40
        # Cost function parameters
        r_eps = 0.05
        # Phase 2 cost function parameters
        lambdas_2 = [1, 1.2, 0]
        # Phase 3 cost function parameters
        lambdas_3 = [1, 0.01, 0.6]

        Y,hist = thesne.tsnet(
            d, output_dims=2, random_state=1, perplexity=40, n_epochs=n,
            Y=None,
            initial_lr=50, final_lr=50, lr_switch=n // 2,
            initial_momentum=momentum, final_momentum=momentum, momentum_switch=n // 2,
            initial_l_kl=lambdas_2[0], final_l_kl=lambdas_3[0], l_kl_switch=n // 2,
            initial_l_c=lambdas_2[1], final_l_c=lambdas_3[1], l_c_switch=n // 2,
            initial_l_r=lambdas_2[2], final_l_r=lambdas_3[2], l_r_switch=n // 2,
            r_eps=r_eps, autostop=tolerance, window_size=window_size,
            verbose=True, a=a
        )

        X = layout_io.normalize_layout(Y)
        stress.append(get_stress(X,d_norm))
        NP.append(get_neighborhood(X,d))

        draw(G,X,output='drawings/random_graphs/linear_comb/{}_a{}.png'.format(graph,a))


    return np.array(NP),np.array(stress)

def get_zeros(n):
    return np.zeros(n,dtype=float)


#Function that takes a graph, embedding function, parameter range, num_params, num_repeats
#Returns list of NP, stress, and draws at drawings/func_name/num_params.png
def embedding(G,matrices, f,g_name,num_params=5, num_repeats=5,c_ids=None,state=None):

    d,d_norm = matrices


    NE, stress, times = get_zeros(num_params), get_zeros(num_params), get_zeros(num_params)
    m1,m2,m3,m4 = [get_zeros(num_params) for _ in range(4)]
    for _ in range(num_repeats):
        start = time.perf_counter()
        Xs, output = f(d,d_norm,num_params,g_name,G)
        end = time.perf_counter()

        NE += np.array( [get_neighborhood(X,d) for X in Xs] )
        stress += np.array([get_stress(X,d) for X in Xs])
        times += (end-start)/len(Xs)

        m = np.array([compute_graph_cluster_metrics(G,X,c_ids) for X in Xs])
        m1 += m[:,0]
        m2 += m[:,1] 

        for X in Xs:
            pos = G.new_vp("vector<float>")
            pos.set_2d_array(X.T)
            state.draw(pos=pos,output=output)

    
    NE /= num_repeats 
    stress /= num_repeats
    times /= num_repeats
    m1 /= num_repeats
    m2 /= num_repeats
    m3 /= num_repeats
    m4 /= num_repeats


    return NE,stress, times,m1,m2,m3,m4

def get_ds(G):
    return distance_matrix.get_distance_matrix(G,'spdm',normalize=False), distance_matrix.get_distance_matrix(G,'spdm')


def experiment(n=5):
    import os
    import pickle
    import copy

    path = 'table_graphs/'
    graph_paths = os.listdir(path)

    # graph_paths = list( map(lambda s: s.split('.')[0], graph_paths) )
    graphs = [(gt.load_graph(f"{path+graph}"),graph) for graph in graph_paths]
    graphs = sorted(graphs,key=lambda x: x[0].num_vertices())
    graph_paths = [g for _,g in graphs]
    print(graph_paths)


    e_funcs = {
        "transformation": embed_l2g_transformation, 
        "all_pairs_log": l2g_rep_all_pairs,
        "comp_pairs_log": l2g_rep_some_pairs,
        "all_pairs_no_log": l2g_rep_all_pairs_nolog,
        "comp_pairs_no_log": l2g_rep_some_pairs_nolog, 
        "tsne": embed_tsne,
        "mds": embed_mds
    }

    data = {
        e: {g: {"NE": None, "stress": None} for g in graph_paths}
        for e in e_funcs.keys()
    }

    import pickle

    for graph in tqdm(graph_paths):
        G = gt.load_graph(f"{path+graph}")
        c_ids, state = get_cluster_ids(G)
        d,d_norm = get_ds(G)
        for f_name, f in e_funcs.items():
            NE, stress,times,m1,m2 = embedding(G,(d,d_norm),f,graph,c_ids,state)
            data[f_name][graph]["NE"] = 1-NE
            data[f_name][graph]["stress"] = stress
            data[f_name][graph]["time"] = times
            data[f_name][graph]["m1"] = m1
            data[f_name][graph]["m2"] = m2
    
            filehandler = open("02_23_test.pkl", 'wb') 
            pickle.dump(data, filehandler)
            filehandler.close()


def test():
    from graph_metrics import apsp
    G = gt.load_graph("graphs/12square.dot")
    prop = G.new_vp("int",vals=[i ** 2 for i in range(G.num_vertices())])

    G.vertex_properties["myprop"] = prop
    return G
    

if __name__ == '__main__':
    experiment(n=5)