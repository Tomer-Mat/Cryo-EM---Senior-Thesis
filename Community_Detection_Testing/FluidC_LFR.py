import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm
from sklearn.metrics.cluster import normalized_mutual_info_score

#Set up LFR graph
n = 100
tau1 = 3
tau2 = 1.5
mu = 0.1
G = nx.LFR_benchmark_graph(n, tau1, tau2, mu, min_degree=5, min_community=25, seed=10)

#Extract community labels for NMI score
set_comm = {frozenset(G.nodes[v]["community"]) for v in G}
comm_list = [node_set for node_set in set_comm]
color_map = list(range(0, 100))

for i, comm in enumerate(comm_list):
    for node_comm in comm:
        color_map[node_comm] = i

ground_labels = color_map

#Extract nodes list to align with community labels
node_list = []
for node_group in comm_list:
    for node in node_group:
        node_list.append(node)
node_list = sorted(node_list)
pos = nx.spring_layout(G)

#Plot LFR graph
plt.figure(1)
plt.title("LFR generated graph")
nx.draw(G, pos, nodelist=node_list, node_color=color_map, cmap=plt.cm.get_cmap('rainbow'), with_labels=True, font_color='white')

#Set up Fluid partition
partition = list(nx_comm.asyn_fluidc(G, k=3, seed=2))

#Extract nodes and community labels lists
node_list_f = sorted(list(partition[0]) + list(partition[1]) + list(partition[2]))
fluid_labels = []
for node in node_list_f:
    if(node in list(partition[0])):
        fluid_labels.append(0)
    elif(node in list(partition[1])):
        fluid_labels.append(1)
    else:
        fluid_labels.append(2)

#Calculate NMI
nmi = normalized_mutual_info_score(ground_labels, fluid_labels)

#Plot Fluid graph
plt.figure(2)
plt.title("Fluid partition, NMI = {}".format(nmi))
nx.draw(G, pos, nodelist=node_list_f, node_color=fluid_labels, cmap=plt.cm.get_cmap('rainbow'), with_labels=True, font_color='white')

plt.show()