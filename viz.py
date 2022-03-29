import numpy as np
import matplotlib.pyplot as plt
import pickle


n = 5

#1f77b4', '#aec7e8', '#ff7f0e'

with open('push_data/lg_random_graphs1.pkl', 'rb') as myfile:
    data = pickle.load(myfile)

with open('push_data/sgd_random_graphs1.pkl', 'rb') as myfile:
    sgd = pickle.load(myfile)

with open('push_data/tsnet_random_graphs1.pkl', 'rb') as myfile:
    tsnet = pickle.load(myfile)

graph = 'block_model300'
NP = data[graph]['NP']
stress = data[graph]['stress']

mds_col = '#bd2d17'
tsnet_col = '#098017'
NP_col = '#ff7f0e'
stress_col = '#1f77b4'

X = [int(i) for i in np.linspace(10,100,8)]

cat = np.concatenate((NP,stress))
bottom = 0
top = 1

fig, ax1 = plt.subplots()
fig.suptitle(graph)
ax1.plot(X, NP, 'o-',color=NP_col,label='NP')
ax1.set_ylabel('NP')#, color='#ff7f0e')
#ax1.tick_params('y', colors='#ff7f0e')

ax1.plot(X,np.ones(stress.shape)*sgd[graph]['NP'],'--',color=mds_col,label='MDS')
ax1.plot(X,np.ones(stress.shape)*tsnet[graph]['NP'],'--',color=tsnet_col,label='tsNET')

#ax1.set_ylim(bottom,top)
fig.legend()
plt.show()
plt.clf()


#ax2 = ax1.twinx()
ax1.plot(X, stress, 'o-',color='#1f77b4',label='stress')
ax1.plot(X,np.ones(stress.shape)*sgd[graph]['stress'],'+--',color='#1f77b4',label='MDS')
ax1.plot(X,np.ones(stress.shape)*tsnet[graph]['stress'],'^--',color='#1f77b4')

ax1.set_xlabel('k')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('stress')#, color='#1f77b4')
ax1.set_ylim(bottom,top)
#ax2.tick_params('y', colors='#1f77b4')



fig.legend()

#fig.tight_layout()
plt.show()

plt.clf()

plt.plot(stress,NP,'o-',label='local-global')
plt.plot(sgd[graph]['stress'],sgd[graph]['NP'],'o',label='MDS')
plt.plot(tsnet[graph]['stress'],tsnet[graph]['NP'],'o',label='tsNET')
plt.xlabel('NP')
plt.ylabel('stress')
plt.xlim(0,1)
plt.ylim(0,1)
plt.legend()

plt.show()
