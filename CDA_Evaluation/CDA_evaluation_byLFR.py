from cdlib import algorithms as cd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np

def generate_lfr(mixing_param):
    return nx.LFR_benchmark_graph(n=N, tau1=3, tau2=1.1, mu=mixing_param,
                                  average_degree=10, max_degree=50, min_community=10, max_community=50, seed=10)
    #return nx.LFR_benchmark_graph(n=100, tau1=3, tau2=1.1, mu=mixing_param, min_degree=5, min_community=25, seed=10)

def extract_communities_list(communities_list):
    community_labels = np.zeros(N)
    for comm_index, comm_nodes in enumerate(communities_list):
        for node in comm_nodes:
            community_labels[node] = comm_index
    return community_labels


N = 1000 #number of nodes in graph
#Initialize NMI values lists
nmi_infomap, nmi_eigenvector, nmi_louvian, \
nmi_leiden, nmi_walktrap, nmi_markov, nmi_greedy, nmi_propagation = ([] for i in range(8))

for mixing_parameter in np.arange(0.1, 1, 0.1):
    ############################### LFR ###############################
    LFR_G = generate_lfr(mixing_parameter)
    set_comm = {frozenset(LFR_G.nodes[v]["community"]) for v in LFR_G}
    comm_list = [node_set for node_set in set_comm]
    true_labels = extract_communities_list(comm_list)
    nx.draw(LFR_G, nx.spring_layout(LFR_G), node_color=true_labels, cmap=plt.cm.get_cmap('rainbow'), node_size=30)
    comm_num = len(true_labels)
    plt.title('len: %i', comm_num)
    plt.show()

    ############################### Infomap ###############################
    infomap_partition = cd.infomap(LFR_G)  # Partition graph with Infomap
    infomap_labels = extract_communities_list(infomap_partition.communities)
    nmi_infomap.append(normalized_mutual_info_score(true_labels, infomap_labels))

    ############################### Leading Eigenvector ###############################
    eigenvector_partition = cd.eigenvector(LFR_G)
    eigenvector_labels = extract_communities_list(eigenvector_partition.communities)
    nmi_eigenvector.append(normalized_mutual_info_score(true_labels, eigenvector_labels))

    ############################### Louvian ###############################
    louvian_partition = cd.louvain(LFR_G)
    louvian_labels = extract_communities_list(louvian_partition.communities)
    nmi_louvian.append(normalized_mutual_info_score(true_labels, louvian_labels))

    ############################### Leiden ###############################
    leiden_partition = cd.leiden(LFR_G)
    leiden_labels = extract_communities_list(leiden_partition.communities)
    nmi_leiden.append(normalized_mutual_info_score(true_labels, louvian_labels))

    ############################### Walktrap ###############################
    walktrap_partition = cd.walktrap(LFR_G)
    walktrap_labels = extract_communities_list(walktrap_partition.communities)
    nmi_walktrap.append(normalized_mutual_info_score(true_labels, walktrap_labels))

    ############################### Markov Clustering ###############################
    markov_partition = cd.markov_clustering(LFR_G)
    markov_labels = extract_communities_list(markov_partition.communities)
    nmi_markov.append(normalized_mutual_info_score(true_labels, markov_labels))

    ############################### Greedy ###############################
    greedy_partition = cd.greedy_modularity(LFR_G)
    greedy_labels = extract_communities_list(greedy_partition.communities)
    nmi_greedy.append(normalized_mutual_info_score(true_labels, greedy_labels))

    ############################### Label Propagation ###############################
    propagation_partition = cd.label_propagation(LFR_G)
    propagation_labels = extract_communities_list(propagation_partition.communities)
    nmi_propagation.append(normalized_mutual_info_score(true_labels, propagation_labels))

#Plot NMI scores
nmi_graph = plt.gca()
nmi_graph.set_xlim([0, 0.9])
nmi_graph.set_ylim([-0.1, 1])
nmi_graph.plot(np.arange(0.1, 1, 0.1), nmi_infomap, color='#575757',
               marker='o', mfc='#f1362b', mec='#f1362b', label="Infomap")
nmi_graph.plot(np.arange(0.1, 1, 0.1), nmi_eigenvector, color='#575757',
               marker='o', mfc='#17c436', mec='#17c436', label="Eigenvector")
nmi_graph.plot(np.arange(0.1, 1, 0.1), nmi_leiden, color='#575757',
               marker='o', mfc='#d9a000', mec='#d9a000', label="Leiden")
nmi_graph.plot(np.arange(0.1, 1, 0.1), nmi_walktrap, color='#575757',
               marker='o', mfc='#532ce9', mec='#532ce9', label="Walktrap")
nmi_graph.plot(np.arange(0.1, 1, 0.1), nmi_louvian, color='#575757',
               marker='x', mfc='#f1362b', mec='#f1362b', label="Louvian")
nmi_graph.plot(np.arange(0.1, 1, 0.1), nmi_markov, color='#575757',
               marker='x', mfc='#17c436', mec='#17c436', label="Markov")
nmi_graph.plot(np.arange(0.1, 1, 0.1), nmi_greedy, color='#575757',
               marker='x', mfc='#d9a000', mec='#d9a000', label="Greedy")
nmi_graph.plot(np.arange(0.1, 1, 0.1), nmi_propagation, color='#575757',
               marker='x', mfc='#532ce9', mec='#532ce9', label="LPA")
plt.ylabel("NMI")
plt.xlabel(r'Mixing Parameter $\mu$')
plt.legend()
plt.grid()
plt.show()