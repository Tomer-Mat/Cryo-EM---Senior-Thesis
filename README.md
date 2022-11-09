# Community Detection with MRA applications - Senior Thesis
## High-Level Description ##
My senior thesis in Electrical Engineering (Detecting similar 1-D Cryo-EM samples at different levels of SNR), graded 96. The project was done in pairs, under the guidance of Dr. Tamir Bendori, from Tel Aviv University. <br>
In this project we tackled the problem of detecting similar (1-D) Cryogenic Electron Microscopy (Cryo-EM) samples, which were represented as (1-D) noisy signals under the Multi Reference Alignment (MRA) model, at different levels of SNR (Signal-to-Noise Ratio). <br> A common solution for this problem is to use the variants of the well-known K-Means algorithm, which yield excellent results and can be computed efficiently. <br> In this project we examined a different method to approach this problem based on Community Detection (CD) algorithms, which are commonly used in Network Analysis and Graph Theory. <br> We showed that under certain conditions our approach yields similar results to the traditional approach, and even better results on other conditions. 

## The MRA Model ##
The Cryo-EM samples are typically modeled by different 1-D signals, which are circularly shifted by an unknown factor, and then are added (independent) Gaussian white noise. Formally, given a set of $k$ 1-D signals $x_1, x_2, ..., x_k \subseteq \mathbb{R}^L$ with known length $L$, we compute for each $x_i$: $$y_i=R_s(x_i)+ Z $$  where $R_s(\cdot)$ is a circular (Random) shift and $Z \sim N(0,\sigma^2)$ <br>
When we say that the problem is heterogenic we mean that the $k$ signals are different from eachother, and that each signal is (circularly) shifted differently. <br>
Using the above terminology, we can restate the problem as follows: Given a set of (1-D) $m$ signals (samples) $y_1, y_2,...,y_m\subseteq \mathbb{R}^L$ created by the transformation above, can we determine which samples are of the same signal? 

## Examined Approach ##
Given the $m$ samples $y_1, y_2,...,y_m\subseteq \mathbb{R}^L$, we applied the following method: Firstly, we built a weighted, complete graph, in which each sample is represented by a vertex and each edge is weighted according to a similarity function (chosen to be the cross-correlation function, which
can be computed fast by FFT). After the graph is built, the similar samples can be found by running a community detection (CD) algorithm.

## More ##
The project is done in pairs, and was written in Python. Our code uses the libaries Numpy, Scipy, Igraph and Cdlib. <br> The project was graded highly, as it produced good results and shown a great skill of self learning and teamwork.
