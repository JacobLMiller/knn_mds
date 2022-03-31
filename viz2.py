import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt
import pickle

import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix

norm = lambda x: np.linalg.norm(x,ord=2)

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

def dist(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += abs(np.linalg.norm(X[i]-X[j])-d[i][j])/d[i][j]
    return stress/choose(len(X),2)

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
            k = len(k_theory[i])
            if k_theory[i][j] in k_embedded[i]:
                count_intersect += 1
        sum += count_intersect/(2*k - count_intersect)

    return sum/len(X)

def stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow(np.linalg.norm(X[i]-X[j])-d[i][j],2)
    return pow(stress,0.5)

with open('data/lg_table_full.pkl', 'rb') as myfile:
    data = pickle.load(myfile)

with open('push_data/sgd_table_graphs.pkl','rb') as myfile:
    sgd_data = pickle.load(myfile)

with open('data/tsnet_table_full.pkl','rb') as myfile:
    tsnet_data = pickle.load(myfile)

print(data.keys())



metric = 'NP'






# plt.plot([1 for _ in range(5)],LG_low,'o',label="LG_low")
# plt.plot([2 for _ in range(5)],LG_high,'o',label="LG_high")
#
#
# plt.legend()
# plt.show()
# plt.clf()

row_labels = list(data.keys())
print(data[row_labels[2]].keys())

row_labels.sort()
#row_labels = ['can_96', 'football', 'jazz', 'visbrazil', 'mesh3e1', 'powerlaw300','block_model_300']

graphs = np.array([gt.load_graph("table_graphs/{}.dot".format(row)).num_vertices() for row in row_labels])
sort_graphs = np.argsort(graphs)
row_labels = [row_labels[i] for i in sort_graphs]
row_labels = ['can_96', 'football', 'jazz', 'visbrazil', 'mesh3e1', 'powerlaw300', 'block_model_300', 'connected_watts_300', 'netscience', 'dwt_419', 'oscil', '494_bus', 'block_model_500', 'powerlaw500', 'connected_watts_500', 'qh882', 'connected_watts_1000', 'block_model_1000', 'powerlaw1000', 'dwt_1005', 'btree9', 'CSphd', 'fpga', 'block_2000', 'sierpinski3d', 'EVA']

max_graph = {graph:0 for graph in row_labels}

column_labels = ["tsnet", 'LG,k=22', 'LG,k=100', 'LG,k=|V|','sgd']

table_string = ""
ismin = lambda score,m: "{}".format(round(score,4)) if not m else "{}\\bf {} {}".format('{',round(score,4),'}')

cell_data = []
count_tsnet, count_sgd = 0,0
for row in row_labels:
    if data[row][metric][0] > 0:
        lg_low = data[row][metric][0]
        lg_mid = data[row][metric][1]
        lg_high = data[row][metric][-1]
        if row in sgd_data:
            sgd = sgd_data[row][metric]
        else:
            sgd = 0
        if row in tsnet_data:
            tsnet = tsnet_data[row][metric]
        else:
            tsnet = 0
        r = lambda x: round(x,4)
        this_row = np.array([r(tsnet),r(lg_low),r(lg_mid),r(lg_high),r(sgd)])
        cell_data.append(this_row)
        min_row = np.argmin(this_row)
        split = row.split("_")
        format = [b + "\_" for b in split[:-1] ] + [split[-1]] if len(split) > 1 else split
        format = "".join(format)
        row_string = "{graph} & {t_score} & {L_s} & {L_m} & {L_l} & {M} \\\ \n".format(graph=format, t_score=ismin(tsnet,0==min_row), L_s=ismin(lg_low,1==min_row), L_m=ismin(lg_mid,2==min_row),L_l=ismin(lg_high,3==min_row), M=ismin(sgd,4==min_row))
        print(row_string)
        table_string += row_string + '\\hline \n'
    else:
        cell_data.append([0,0,0,0])

print(table_string)
print("low performed better on ", count_tsnet)
print("hgih performed better on ", count_sgd)

plt.suptitle("NP")
table = plt.table(cellText=cell_data,
                      rowLabels=row_labels,
                      colLabels=column_labels,
                      loc='center',
                      colWidths=[0.2,0.2,0.2,0.2,0.2],
                      cellLoc='center')

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

for (row, col), cell in table.get_celld().items():
    # print(row)
    # print(row_labels[row-1])
    # print()
    if row == 0: continue
    if (max_graph[row_labels[row-1]] == col):
        cell.set_text_props(fontproperties=FontProperties(weight='bold'))

plt.xticks([])
plt.yticks([])
plt.axis('off')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

#plt.show()
#plt.savefig('prelim_table_stress.eps')

plt.clf()



import matplotlib.cm as cm

#
def plot_metric_table(metric):
    fig, ax = plt.subplots()
    rows = row_labels
    columns = ["tsnet", 'LG,k=22', 'LG,k=100', 'LG,k=|V|','sgd']

    conf_data = np.array(cell_data)

    colores = np.zeros((conf_data.shape[0],conf_data.shape[1],4))
    for i in range(conf_data.shape[0]):
        col_data = conf_data[i]
        normal = plt.Normalize(np.min(col_data), np.max(col_data))
        colores[i,:] = cm.RdYlGn_r(normal(col_data))

    #fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    mytable = ax.table(cellText=conf_data,
             rowLabels=rows,
             colLabels=columns,
             cellColours=colores,
             loc='center',
             colWidths=[0.1 for x in columns])
    fig.tight_layout()

    #mytable.set_fontsize(20)


    plt.savefig('{}_table.png'.format(metric))

def plot_time():
    fig, ax = plt.subplots()
    cell_data = np.zeros((len(row_labels),1))

    for i,graph in zip(range(len(row_labels)),row_labels):
        print(data[graph]['time'])
        cell_data[i] = round(data[graph]['time'].mean(),3)


    ax.axis('off')
    ax.axis('tight')
    mytable = ax.table(cellText=cell_data,
                        rowLabels=row_labels,
                        colLabels=['Time (s)'],
                        loc='center',
                        colWidths=[0.1])
    fig.tight_layout()

    plt.savefig('time_table.png')

# plot_metric_table('NP')
# plot_metric_table('stress')
# plot_time()
