# Dataset used in the following code:
# M. Girvan and M. E. J. Newman, Community structure in social and biological networks,
#Proc. Natl. Acad. Sci. USA 99, 7821-7826 (2002).

# Variation of information (VI)
#
# Meila, M. (2007). Comparing clusterings-an information
#   based distance. Journal of Multivariate Analysis, 98,
#   873-895. doi:10.1016/j.jmva.2006.11.013
#   Code in function variation_of_information is from: https://gist.github.com/jwcarr/626cbc80e0006b526688
#
# https://en.wikipedia.org/wiki/Variation_of_information

from cdlib import algorithms as cd
from cdlib import evaluation as eval
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import time
from math import log

def extract_communities_list(communities_list, G):
    community_labels = np.zeros(N)
    for comm_index, comm_nodes in enumerate(communities_list):
        for node in comm_nodes:
            community_labels[list(G).index(node)] = comm_index
    return community_labels

def get_partitions(communities_list):
    partitions =  [[] for i in range((int(max(communities_list)) + 1))] #create a list of C lists (C=number of communities)
    for node_index, community in enumerate(communities_list):
        partitions[int(community)].append(node_index)
    return partitions

def variation_of_information(X, Y):
  n = float(sum([len(x) for x in X]))
  sigma = 0.0
  for x in X:
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n
      if r > 0.0:
        sigma += r * (log(r / p, 2) + log(r / q, 2))
  return abs(sigma)

gml_file = open("football.gml", "r")
#gml_file = open("polbooks.gml", "r")

gml_list = []
for line in gml_file:
    gml_list.append(line)
gml_list = [item.replace('\n', '') for item in gml_list]

G = nx.parse_gml(gml_list)
N = G.number_of_nodes()
dt_g = nx.get_node_attributes(G, 'value')
true_communities = list(dt_g.values())
true_partitions = get_partitions(true_communities)

plt.figure()
plt.title("Amazon books labeled by political map (right, left, neutral)")
nx.draw(G, nx.spring_layout(G), node_color=true_communities, cmap=plt.cm.get_cmap('brg'), node_size=70)
plt.show()

############################### Infomap ###############################
start_time = time.time()
infomap_partition = cd.infomap(G)
infomap_time = time.time() - start_time
infomap_communities = extract_communities_list(infomap_partition.communities, G)
infomap_partitions = get_partitions(infomap_communities)
nmi_infomap = normalized_mutual_info_score(true_communities, infomap_communities)
ari_infomap = adjusted_rand_score(true_communities, infomap_communities)
vi_infomap = variation_of_information(true_partitions, infomap_partitions)

############################### Leading Eigenvector ###############################
start_time = time.time()
eigenvector_partition = cd.eigenvector(G)
eifenvector_time = time.time() - start_time
eigenvector_communities = extract_communities_list(eigenvector_partition.communities, G)
eigenvector_paritions = get_partitions(eigenvector_communities)
nmi_eigenvector = normalized_mutual_info_score(true_communities, eigenvector_communities)
ari_eigenvector = adjusted_rand_score(true_communities, eigenvector_communities)
vi_eigenvector = variation_of_information(true_partitions, eigenvector_paritions)

############################### Louvian ###############################
start_time = time.time()
louvian_partition = cd.louvain(G)
louvian_time = time.time() - start_time
louvian_communities = extract_communities_list(louvian_partition.communities, G)
louvian_partitions = get_partitions(louvian_communities)
nmi_louvian = normalized_mutual_info_score(true_communities, louvian_communities)
ari_louvian = adjusted_rand_score(true_communities, louvian_communities)
vi_louvian = variation_of_information(true_partitions, louvian_partitions)

############################### Leiden ###############################
start_time = time.time()
leiden_partition = cd.leiden(G)
leiden_time = time.time() - start_time
leiden_communities = extract_communities_list(leiden_partition.communities, G)
leiden_partitions = get_partitions(leiden_communities)
nmi_ledien = normalized_mutual_info_score(true_communities, leiden_communities)
ari_leiden = adjusted_rand_score(true_communities, leiden_communities)
vi_leiden = variation_of_information(true_partitions, leiden_partitions)

############################### Walktrap ###############################
start_time = time.time()
walktrap_partition = cd.walktrap(G)
walktrap_time = time.time() - start_time
walktrap_communities = extract_communities_list(walktrap_partition.communities, G)
walktrap_partitions = get_partitions(walktrap_communities)
nmi_walktrap = normalized_mutual_info_score(true_communities, walktrap_communities)
ari_walktrap = adjusted_rand_score(true_communities, walktrap_communities)
vi_walktrap = variation_of_information(true_partitions, walktrap_partitions)

############################### Markov Clustering ###############################
start_time = time.time()
markov_partition = cd.markov_clustering(G)
markov_time = time.time() - start_time
markov_communities = extract_communities_list(markov_partition.communities, G)
markov_partitions = get_partitions(markov_communities)
nmi_markov = normalized_mutual_info_score(true_communities, markov_communities)
ari_markov = adjusted_rand_score(true_communities, markov_communities)
vi_markov = variation_of_information(true_partitions, markov_partitions)

############################### Greedy ###############################
start_time = time.time()
greedy_partition = cd.greedy_modularity(G)
greedy_time = time.time() - start_time
greedy_communities = extract_communities_list(greedy_partition.communities, G)
greedy_partitions = get_partitions(greedy_communities)
nmi_greedy = normalized_mutual_info_score(true_communities, greedy_communities)
ari_greedy = adjusted_rand_score(true_communities, greedy_communities)
vi_greedy = variation_of_information(true_partitions, greedy_partitions)

############################### Label Propagation ###############################
start_time = time.time()
propagation_partition = cd.label_propagation(G)
propagation_time = time.time() - start_time
propagation_communities = extract_communities_list(propagation_partition.communities, G)
propagation_partitions = get_partitions(propagation_communities)
nmi_propagation = normalized_mutual_info_score(true_communities, propagation_communities)
ari_propagation = adjusted_rand_score(true_communities, propagation_communities)
vi_propagation = variation_of_information(true_partitions, propagation_partitions)


nmi_x = np.arange(8)
ari_x = [x + 0.3 for x in nmi_x]
vi_x = [x + 0.3 for x in ari_x]

#Plot NMI score
plt.grid(zorder=0)
plt.bar(nmi_x,
        [nmi_infomap, nmi_eigenvector, nmi_louvian, nmi_ledien, nmi_walktrap, nmi_markov, nmi_greedy, nmi_propagation],
         width=0.3,label="NMI",color='#266ed9',zorder=3)
plt.bar(ari_x,
        [ari_infomap, ari_eigenvector, ari_louvian, ari_leiden, ari_walktrap, ari_markov, ari_greedy, ari_propagation],
         width=0.3,label="ARI",color='#ef102d',zorder=3)
plt.bar(vi_x,
         [vi_infomap, vi_eigenvector, vi_louvian, vi_leiden, vi_walktrap, vi_markov, vi_greedy, vi_propagation],
         width=0.3,label="VI",color='#42bd56',zorder=3)
plt.xticks(ticks=[x + 0.3 for x in range(8)],
           labels=["Infomap", "Eigenvector", "Louvian", "Leiden", "Walktrap", "Markov", "Greedy", "LPA"],
           rotation=45, ha="right")
plt.legend()
plt.title("Algorithms scores for the football games dataset")
plt.show()

#Plot time
plt.grid(zorder=0)
plt.bar(["Infomap", "Eigenvector", "Louvian", "Leiden", "Walktrap", "Markov", "Greedy", "LPA"],
         [infomap_time, eifenvector_time, louvian_time, leiden_time, walktrap_time, markov_time, greedy_time, propagation_time],
         color='#de8221',zorder=3)
plt.ylabel("Time[s]")
plt.title("Algorithms execution time for the football games dataset")
plt.xticks(rotation=45, ha="right")
plt.show()