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

graph = 'connected_watts_700'
NP = data[graph]['NP']
stress = data[graph]['stress']

X = [i for i in range(len(NP))]

cat = np.concatenate((NP,stress))
bottom = 0
top = 1

fig, ax1 = plt.subplots()
ax1.plot(X, NP, 'o-',color='#ff7f0e',label='NP')
ax1.set_ylabel('NP', color='#ff7f0e')
ax1.tick_params('y', colors='#ff7f0e')
ax1.set_ylim(bottom,top)


ax2 = ax1.twinx()
ax2.plot(X, stress, 'o-',color='#1f77b4',label='stress')
ax2.plot(X,np.ones(stress.shape)*sgd[graph]['stress'],'--',color='green',label='MDS')
ax2.set_xlabel('k')
# Make the y-axis label, ticks and tick labels match the line color.
ax2.set_ylabel('stress', color='#1f77b4')
ax2.set_ylim(bottom,top)
ax2.tick_params('y', colors='#1f77b4')



fig.legend()

#fig.tight_layout()
plt.show()

plt.clf()

plt.plot(NP,stress,'o-',label='local-global')
plt.plot(sgd[graph]['NP'],sgd[graph]['stress'],'o',label='MDS')
plt.plot(tsnet[graph]['NP'],tsnet[graph]['stress'],'o',label='tsNET')
plt.xlabel('NP')
plt.ylabel('stress')
plt.xlim(0,1)
plt.ylim(0,1)
plt.legend()

plt.show()
