# Cryo-EM---Senior-Thesis
My senior thesis in Electrical Engineering (Detecting similar 1-D Cryo-EM samples at different levels of SNR), graded 96.
In this project we tackled the problem of detecting similar (1-D) Cryogenic Electron Microscopy (Cryo-EM) samples at different levels of SNR (Signal-to-Noise Ratio).
The Cryo-EM samples are typically modeled by different 1-D signals, which are circularly shifted by an unknown factor, and then are added (independent) Gaussian white noise.
A common approach to solve this problem is to use variants of the famous K-means algorithm, which yields fast and satisfying results. In our project a different approach is proposed: first,
we build a weighted, complete graph, in which each sample is represented by a vertex and each edge is weighted according to a similarity function (chosen to be the cross-correlation function, which
can be computed fast by FFT). After the graph is built, the similar samples can be found by running a community detection (CD) algorithm.
In our senior thesis, we compared the performance of the traditional approach with the performance of various CD algorithms (Edge Betweennes, Fast Greedy, Label Propogation, Louvian, etc) based on the Normalized Mutual Information (NMI) metric and the computational complexity. We showed that under certein conditions, our proposed approach yields better results than the traditional approach.
The project is written in Python; The well-known libaries Numpy, Scipy, Igraph and Cdlib are also used.
