import numpy as np
import matplotlib.pyplot as plt
import pickle


n = 5

#1f77b4', '#aec7e8', '#ff7f0e'

with open('data/paramater_experiments.pkl', 'rb') as myfile:
    data = pickle.load(myfile)

with open('data/lg_table_full.pkl', 'rb') as myfile:
    full = pickle.load(myfile)

with open('push_data/sgd_table_graphs.pkl', 'rb') as myfile:
    sgd = pickle.load(myfile)

with open('data/tsnet_table_full.pkl', 'rb') as myfile:
    tsnet = pickle.load(myfile)

graph = 'block_model_500'
mds_col = '#bd2d17'
tsnet_col = '#098017'
NP_col = '#ff7f0e'
stress_col = '#1f77b4'

# for graph in data.keys():
#
#     NP = np.array( [x for x in data[graph]['NP']] )#+ [full[graph]['NP'][-1]] )
#
#     stress = np.array( [x for x in data[graph]['stress']])# + [full[graph]['stress'][-1]] )
#
#     mds_col = '#bd2d17'
#     tsnet_col = '#098017'
#     NP_col = '#ff7f0e'
#     stress_col = '#1f77b4'
#
#     X = [int(i) for i in np.linspace(10,100,8)]
#
#     #cat = np.concatenate((NP,stress))
#     bottom = 0
#     top = 1
#
#     fig, ax1 = plt.subplots()
#     fig.suptitle(graph)
#     ax1.plot(X, NP, 'o-',color=NP_col,label='NE')
#     ax1.set_ylabel('NE')#, color='#ff7f0e')
#     #ax1.tick_params('y', colors='#ff7f0e')
#
#     ax1.plot(X,np.ones(stress.shape)*sgd[graph]['NP'],'--',color=mds_col,label='MDS')
#     ax1.plot(X,np.ones(stress.shape)*tsnet[graph]['NP'],'--',color=tsnet_col,label='tsNET')
#
#     #ax1.set_ylim(bottom,top)
#     fig.legend()
#     plt.savefig('figures/new_kcurve/{}_1.png'.format(graph))
#     plt.close()
#
#
#
#     fig, ax1 = plt.subplots()
#     fig.suptitle(graph)
#     ax1.plot(X, stress, 'o-',color=stress_col,label='stress')
#     ax1.set_ylabel('stress')#, color='#ff7f0e')
#     #ax1.tick_params('y', colors='#ff7f0e')
#
#     ax1.plot(X,np.ones(stress.shape)*sgd[graph]['stress'],'--',color=mds_col,label='MDS')
#     ax1.plot(X,np.ones(stress.shape)*tsnet[graph]['stress'],'--',color=tsnet_col,label='tsNET')
#
#     #ax1.set_ylim(bottom,top)
#     fig.legend()
#     plt.savefig('figures/new_kcurve/{}_2.png'.format(graph))
#     plt.close()
#
#
#     plt.plot(NP,stress,'o-',label='local-global')
#     plt.plot(sgd[graph]['NP'],sgd[graph]['stress'],'o',label='MDS')
#     plt.plot(tsnet[graph]['NP'],tsnet[graph]['stress'],'o',label='tsNET')
#     plt.xlabel('NE')
#     plt.ylabel('stress')
#     plt.suptitle(graph)
#     # plt.xlim(0,1)
#     # plt.ylim(0,1)
#     plt.legend()
#
#     plt.savefig('figures/new_kcurve/{}_3.png'.format(graph))
#     plt.close()

for graph in data.keys():
    print(data[graph].keys())
    NP = data[graph]['matrix_power']['NP']

    stress = data[graph]['matrix_power']['stress']# + [full[graph]['stress'][-1]] )

    print(NP)
    all_data = np.concatenate((NP,stress))
    bottom = np.min(all_data) - 0.01
    top = np.max(all_data) + 0.01
    K = np.linspace(22,100,8)

    plt.suptitle(graph)
    ax1 = plt.subplot()
    l1, = ax1.plot(K,NP, 'o-',color=NP_col)
    ax1.set_ylabel('NE')
    ax1.set_xlabel('k')
    ax1.set_ylim(bottom,top)

    ax2 = ax1.twinx()
    l2, = ax2.plot(K,stress, 'o-',color=stress_col)
    ax2.set_ylabel('stress')
    ax2.set_ylim(bottom,top)

    plt.legend([l1, l2], ["NE","stress"])

    plt.savefig('figures/dual_k_curve/{}_dual.eps'.format(graph))
    plt.clf()
    plt.close()
