import numpy as np
from scipy import signal
import Generate
from scipy import stats

def MRA_Rect_Trian(N, L, K, sigma):
    x = np.zeros((K, L))
    # Generate Rectangle at x[0]
    for l in range(int(L / 4)):
        x[0][l] = 1

    x[0] = (x[0] - np.mean(x[0])) / np.linalg.norm(x[0] - np.mean(x[0]), 2)  # Normalize signal

    # Generate Triangle at x[1]
    x[1] = signal.triang(L)
    x[1] = (x[1] - np.mean(x[1])) / np.linalg.norm(x[1] - np.mean(x[1]), 2)  # Normalize signal

    y, true_partition = Generate.generate_MRA(N, K, L, sigma, x)
    max_corr = Generate.generate_maxcorr(N, L, y)

    G = Generate.generate_graph(max_corr, true_partition)

    return G, true_partition

def MRA_StandardNormal(N, L, K, sigma):
    x = np.zeros((K, L))
    # Generate Standard Normally Distributed signals
    for k in range(K):
        x[k] = np.random.standard_normal(L)
        x[k] = (x[k] - np.mean(x[k])) / np.linalg.norm(x[k] - np.mean(x[k]), 2)  # Normalize signal

    y, true_partition = Generate.generate_MRA(N, K, L, sigma, x)
    max_corr = Generate.generate_maxcorr(N, L, y)

    G = Generate.generate_graph(max_corr, true_partition)

    return G, true_partition

#Generate Correlated signals functions:

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
    return Sigma

# Generate one signal with covariance by known function
def generate_a_signal(L,a,b,choice):
    if(b==1):
        sigi=np.random.randn(L)
    else:
        Sigma=create_covariance_matrix(L,choice,a,b)
        sigi = np.random.multivariate_normal(np.zeros((L,)),Sigma)
    return sigi

def MRA_CorrelatedNormal(N, L, K, a, b, choice, sigma):
    x = np.zeros((K, L))

    # Generate Standard Normally Distributed signals
    for k in range(K):
        x[k] = generate_a_signal(L, a, b, choice)
        x[k] = (x[k] - np.mean(x[k])) / np.linalg.norm(x[k] - np.mean(x[k]), 2) # Normalize signal

    y, true_partition = Generate.generate_MRA(N, K, L, sigma, x)
    max_corr = Generate.generate_maxcorr(N, L, y)

    G = Generate.generate_graph(max_corr, true_partition)

    return G, true_partition
