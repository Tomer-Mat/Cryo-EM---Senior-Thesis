import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import networkx.algorithms.community as nx_comm

#Set up ground truth partition
G = nx.karate_club_graph()
dt_g = nx.get_node_attributes(G,'club')
dt_values_g = list(dt_g.values())
color_map_g = [0 if x == 'Mr. Hi' else 2 for x in dt_values_g]
pos = nx.spring_layout(G)

communities_g = {}
for key, value in sorted(dt_g.items()):
    communities_g.setdefault(value, []).append(key)
communities_g_sets = [set(x) for x in communities_g.values()]
q_g = nx_comm.modularity(G,communities_g_sets)

#Plot ground truth partition
plt.figure(1)
plt.title("Ground truth partition, Q={}".format(q_g))
nx.draw(G, pos, node_color = color_map_g, cmap = plt.cm.get_cmap('rainbow'), with_labels=True, font_color = 'white')

#Set up Louvian partition
partition = community_louvain.best_partition(G)
communities_l = {}
for key, value in sorted(partition.items()):
    communities_l.setdefault(value, []).append(key)
communities_l_sets = [set(x) for x in communities_l.values()]
q_l = nx_comm.modularity(G,communities_l_sets)

#Plot Louvian partition
plt.figure(2)
plt.title("Louvian method partition, Q={}".format(q_l))
nx.draw(G, pos, cmap=plt.cm.get_cmap('rainbow'), node_color=list(partition.values()), with_labels=True, font_color = 'white')

plt.show()