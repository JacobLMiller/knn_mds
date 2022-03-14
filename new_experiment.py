import numpy as np


def main():
    path = 'random_runs/'
    graph_paths = os.listdir(path)

    graph_paths = list( map(lambda s: s.split('.')[0], graph_paths) )
    
    
    for graph in graph_paths:
        G = gt.load_graph(path+graph+'.dot') 
        d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
        d_norm = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)
        
        print("------------------------------")
        print("Graph: {}".format(graph))
        
        K = np.linspace(5,100,8)
        NP = np.zeros(K.shape)
        stress = np.zeros(K.shape)
        
    
    
if __name__ == '__main__':
    main(n=10)
