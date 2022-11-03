import numpy as np
import Generate
import networkx as nx
import matplotlib.pyplot as plt

# Parameters
N = 30 # Number of observations
L = 50 # Signals length
K = 2 # Number of signals
sigma = 0.2 # Noise level

x = np.zeros((K,L))
# Generate Standard Normally Distributed signals
for k in range(K):
    x[k] = np.random.standard_normal(L)
    x[k] = (x[k] - np.mean(x[k]))/np.linalg.norm(x[k] - np.mean(x[k]), 2) # Normalize signal

y, true_partition = Generate.generate_MRA(N, K, L, sigma, x)
max_corr = Generate.generate_maxcorr(N, L, y)

G = Generate.generate_graph(max_corr, true_partition)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
pos = nx.spring_layout(G)
plt.title("Standard Normal Gaussian MRA samples")
nx.draw(G, pos, node_color=true_partition, node_size=20, edgelist=edges, edge_color=weights, width=1, cmap=plt.cm.jet, edge_cmap=plt.cm.Greens)
plt.show()