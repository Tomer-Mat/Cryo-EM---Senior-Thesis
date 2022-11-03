import numpy as np
from scipy import signal
import Generate
import networkx as nx
import matplotlib.pyplot as plt

# Parameters
N = 30 # Number of observations
L = 50 # Signals length
K = 2 # Number of signals
sigma = 0 # Noise level

x = np.zeros((K,L))
# Generate Rectangle at x[0]
for l in range(int(L/4)):
    x[0][l] = 1

x[0] = (x[0] - np.mean(x[0]))/np.linalg.norm(x[0] - np.mean(x[0]), 2) # Normalize signal

# Generate Triangle at x[1]
x[1] = signal.triang(L)
x[1] = (x[1] - np.mean(x[1]))/np.linalg.norm(x[1] - np.mean(x[1]), 2) # Normalize signal

y, true_partition = Generate.generate_MRA(N, K, L, sigma, x)
max_corr = Generate.generate_maxcorr(N, L, y)

G = Generate.generate_graph(max_corr, true_partition)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
pos = nx.spring_layout(G)
plt.title("MRA samples graph. Blue=Rectangle;Red=Triangle")
nx.draw(G, pos, node_color=true_partition, edgelist=edges, edge_color=weights, width=2, cmap=plt.cm.jet, edge_cmap=plt.cm.Greens)
plt.show()