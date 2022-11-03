import numpy as np
import scipy as sc
import Generate
import networkx as nx
import matplotlib.pyplot as plt

# Parameters
N = 100 # Number of observations
L = 51 # Signals length
K = 10 # Number of signals
a = 1 # coefficient
b = 2 # parameter to apply
sigma = 0.2 # Noise level

def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W

def create_arr(L):
    return (np.arange(1, 1+ (L+1) / 2))

def apply_function(arr,choice,a,b): #'1' for a*x^(-b), '2' for a*b^(-x)
    if (choice==1):
        PSD_Oneside=a * (1 / ((arr) ** b))
    else:
       PSD_Oneside= a * (b**((-1)* arr))
    return PSD_Oneside

def create_covariance_matrix(L,choice,a,b): #'1' for a*x^(-b), '2' for a*b^(-x)
    F=DFT_matrix(L)
    arry = create_arr(L)
    PSD_oneside=apply_function(arry,choice,a,b)
    PSD=np.concatenate([PSD_oneside,PSD_oneside[1::][::-1]]) #change here [2::] for even L
    D=np.diag(PSD)
    F_conju=np.conj(F)
    F_transposy=F_conju.T
    Sigma=np.real(F_transposy @ D @ F)
    print(Sigma)
    return Sigma

x = np.zeros((K,L))

# Generate one signal with covariance by known function
def generate_a_signal(L,a,b,choice):
    if(b==1):
        sigi=np.random.randn(L)
    else:
        Sigma=create_covariance_matrix(L,choice,a,b)
        sigi = np.random.multivariate_normal(np.zeros((L,)),Sigma)
    return sigi

def generate_several_signals(L,a,b,choice,K):
    for k in range(K):
        x[k]=generate_a_signal(L,a,b,choice)
        x[k] = (x[k] - np.mean(x[k])) / np.linalg.norm(x[k] - np.mean(x[k]), 2)
    return x

x=generate_several_signals(L,a,b,1,K)

y, true_partition = Generate.generate_MRA(N, K, L, sigma, x)
max_corr = Generate.generate_maxcorr(N, L, y)

G = Generate.generate_graph(max_corr, true_partition)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
pos = nx.spring_layout(G)
plt.title("Standard Normal Gaussian MRA samples")
nx.draw(G, pos, node_color=true_partition, node_size=20, edgelist=edges, edge_color=weights, width=1, cmap=plt.cm.jet, edge_cmap=plt.cm.Greens)
plt.show()