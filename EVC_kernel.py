import numpy as np
import networkx as nx

#compute a list of EV centrality vectors from a list of adjacency matrices
def compute_centralities(A_list):
    centrality_list = list()
    for a in A_list:
        g = nx.from_numpy_array(a)
        val = nx.eigenvector_centrality_numpy(g)
        result = val.items()
        data = list(result)
        val = np.array(data)
        val = val[:, 1]
        centrality_list.append(val)
    return centrality_list

#using these EV centrality vectors use the dot product to compute train and test kernel matrices
def compute_EVC_ttsplit(centralities_train_list, centralities_test_list):

    kernelTrain = np.empty((len(centralities_train_list), len(centralities_train_list)))
    for i in range(len(centralities_train_list)):
        for j in range(len(centralities_train_list)):
            kernelTrain[i, j] = np.dot(centralities_train_list[i], centralities_train_list[j])

    kernelTest = np.empty((len(centralities_test_list), len(centralities_train_list)))
    for i in range(len(centralities_test_list)):
        for j in range(len(centralities_train_list)):
            kernelTest[i, j] = np.dot(centralities_test_list[i], centralities_train_list[j])

    return kernelTrain, kernelTest



