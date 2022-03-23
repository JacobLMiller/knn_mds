import s_gd2
import graph_tool.all as gt
G = gt.load_graph("graphs/dwt_419.dot")
I = []
J = []
for e1,e2 in G.iter_edges():
    I.append(e1)
    J.append(e2)

print('now')
X = s_gd2.layout_convergent(I, J)
print('end')
#s_gd2.draw_svg(X, I, J, 'C5.svg')
print(type(X))
pos = G.new_vp('vector<float>')
pos.set_2d_array(X.T)
gt.graph_draw(G,pos=pos)
