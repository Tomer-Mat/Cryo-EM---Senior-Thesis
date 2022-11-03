from cdlib import algorithms as cd
import MRA_Graphs
import networkx as nx
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def partition_edges_by_communities(edges_list, community_labels):
    intra_edges_weights = []
    inter_edges_weights = []

    for edge in edges_list:
        # if edge connects nodes from same community
        if [True for community in community_labels if (edge[0][0] in community and edge[0][1] in community)]:
            intra_edges_weights.append(edge[1])
        else:
            inter_edges_weights.append(edge[1])

    return intra_edges_weights, inter_edges_weights

def plot_historgram(ax, intra_weights, inter_weights, edges_num, algorithm_name):
    ax.hist(intra_weights, bins=np.arange(11) / 10 + 0.01, alpha=0.5,
             weights=np.ones(len(intra_weights)) / edges_num, label="intra edges weights")
    ax.hist(inter_weights, bins=np.arange(11) / 10 + 0.01, alpha=0.5,
             weights=np.ones(len(inter_weights)) / edges_num, label="inter edges weights")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(algorithm_name)
    ax.legend()

def create_communities_2darray(communities_labels):
    communities_num = len(set(communities_labels))
    communities_array = [[] for x in range(communities_num)]

    for i, label in enumerate(communities_labels):
        communities_array[label].append(i)

    return communities_array


N = 100
K = 2
sigma = 0
G, true_labels = MRA_Graphs.MRA_StandardNormal(N, L=50, K=K, sigma=sigma)

edges_num = G.number_of_edges()
weights_dict = nx.get_edge_attributes(G,'weight')

intra_weights = {"leiden": [], "louvain": [], "leading_eigenvector": [], "multilevel": [], "spinglass": [],
                  "walktrap": []}
inter_weights = {"leiden": [], "louvain": [], "leading_eigenvector": [], "multilevel": [], "spinglass": [],
                  "walktrap": []}

# Execute algorithms and get inter and intra edges weights
leiden = cd.leiden(G, weights='weight')
intra_weights["leiden"], inter_weights["leiden"] = \
    partition_edges_by_communities(weights_dict.items(), leiden.communities)
louvain = cd.louvain(G, weight='weight')
intra_weights["louvain"], inter_weights["louvain"] = \
    partition_edges_by_communities(weights_dict.items(), louvain.communities)
# Convert NetworkX graph to Igraph graph
G = ig.Graph.from_networkx(G)
leading_eigenvector = ig.Graph.community_leading_eigenvector(G, weights="weight")
leading_eigenvector_communities = create_communities_2darray(leading_eigenvector.membership)
intra_weights["leading_eigenvector"], inter_weights["leading_eigenvector"] = \
    partition_edges_by_communities(weights_dict.items(), leading_eigenvector_communities)
multilevel = ig.Graph.community_multilevel(G, weights="weight")
multilevel_communities = create_communities_2darray(multilevel.membership)
intra_weights["multilevel"], inter_weights["multilevel"] = \
    partition_edges_by_communities(weights_dict.items(), multilevel_communities)
spinglass = ig.Graph.community_spinglass(G, weights="weight")
spinglass_communities = create_communities_2darray(spinglass.membership)
intra_weights["spinglass"], inter_weights["spinglass"] = \
    partition_edges_by_communities(weights_dict.items(), spinglass_communities)
walktrap = ig.Graph.community_walktrap(G, weights="weight")
walktrap = walktrap.as_clustering()
walktrap_communities = create_communities_2darray(walktrap.membership)
intra_weights["walktrap"], inter_weights["walktrap"] = \
    partition_edges_by_communities(weights_dict.items(), walktrap_communities)

# Plot histograms
fig, axs = plt.subplots(3, 2)
fig.suptitle("Edges distribution by weight for sigma={0},K={1}".format(sigma, K))

plot_historgram(fig.axes[0], intra_weights["leiden"], inter_weights["leiden"], edges_num, "leiden")
plot_historgram(fig.axes[1], intra_weights["louvain"], inter_weights["louvain"], edges_num, "louvain")
plot_historgram(fig.axes[2], intra_weights["leading_eigenvector"], inter_weights["leading_eigenvector"], edges_num, "leading_eigenvector")
plot_historgram(fig.axes[3], intra_weights["multilevel"], inter_weights["multilevel"], edges_num, "multilevel")
plot_historgram(fig.axes[4], intra_weights["spinglass"], inter_weights["spinglass"], edges_num, "spinglass")
plot_historgram(fig.axes[5], intra_weights["walktrap"], inter_weights["walktrap"], edges_num, "walktrap")

plt.show()