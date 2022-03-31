from SGD_MDS import SGD_MDS, k_nearest_embedded
from MDS_classic import MDS
import graph_tool.all as gt
import time
import numpy as np
import matplotlib.pyplot as plt

import scipy.io
import random
import pickle

norm = lambda x: np.linalg.norm(x,ord=2)



def main():
    with open('data/lg_table_plots.pkl', 'rb') as myfile:
        data_small = pickle.load(myfile)
    with open('data/lg_table_plots_large.pkl', 'rb') as myfile:
        data_large = pickle.load(myfile)

    with open('push_data/sgd_random_graphs1.pkl', 'rb') as myfile:
        sgd = pickle.load(myfile)

    with open('push_data/tsnet_random_graphs1.pkl', 'rb') as myfile:
        tsnet = pickle.load(myfile)

    data = {key: data_small[key] if data_small[key]['NP'][0] != 0 else data_large[key] for key in data_small.keys()}

    print(data_small['connected_watts_300'])

    with open('data/lg_table_plots_full.pkl','wb') as myfile:
        pickle.dump(data,myfile)
    myfile.close()


if __name__ == "__main__":
    main()
