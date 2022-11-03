import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm
from sklearn.metrics.cluster import normalized_mutual_info_score

#Set up ground truth partition
G = nx.karate_club_graph()
dt_g = nx.get_node_attributes(G,'club')
dt_values_g = list(dt_g.values())
ground_labels = [0 if x == 'Mr. Hi' else 2 for x in dt_values_g]
pos = nx.spring_layout(G)

#Plot ground truth partition
plt.figure(1)
plt.title("Ground truth partition")
nx.draw(G, pos, node_color=ground_labels, cmap=plt.cm.get_cmap('rainbow'), with_labels=True, font_color='white')

#Set up Fluid partition
partition = list(nx_comm.asyn_fluidc(G, k=2, seed=2))
node_list = sorted(list(partition[0]) + list(partition[1]))
fluid_labels = [0 if node in list(partition[0]) else 2 for node in node_list]

#Calculate NMI
nmi = normalized_mutual_info_score(ground_labels, fluid_labels)

#Plot Fluid partition
plt.figure(2)
plt.title("Fluid partition, NMI = {}".format(nmi))
nx.draw(G, pos, nodelist=node_list, node_color=fluid_labels, cmap=plt.cm.get_cmap('rainbow'), with_labels=True, font_color='white')

plt.show()