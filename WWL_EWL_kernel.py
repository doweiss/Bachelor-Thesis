import numpy as np
import ot #optimal transport library https://github.com/PythonOT/POT



def assimilate_labels(A_list, node_label_list, depth): #update all of the node labels

    print(len(A_list))
    print(len(node_label_list))

    for iterator in range(len(A_list)):

        A = A_list[iterator]
        node_label = node_label_list[iterator]
        for d in range(depth):
            old_node_label = np.copy(node_label)

            for i in range(A.shape[0]):
                sum = 0 #sum of adjacent nodes
                deg = 0 #degree of current node
                #for every node in the graph
                for j in range(A.shape[0]):
                    if A[i][j] != 0 and i != j:

                        sum = sum + A[i][j] * old_node_label[j][:]
                        sum = sum + old_node_label[j][:]
                        deg = deg + 1
                if deg==0: #to prevent division by zero
                    deg = 1
                node_label[i][:] = 0.5*(old_node_label[i][:] + sum/deg) #make a new node label as the average of the current label

        node_label_list[iterator] = node_label

    return node_label_list



def compute_wasserstein(list1, list2):

    kernelM = np.empty((len(list2), len(list1)))
    print(kernelM.shape)# empty kernel matrix

    for i in range(len(list2)):
        if i % 200 == 0:
            print('Wasserstein:' + str(i + 1) + '/' + str(len(list2)))

        for j in range(len(list1)):
            M = ot.dist(list2[i], list1[j], metric='euclidean')  # all the euclidian ground distances
            u = ot.utils.unif(list2[i].shape[0])
            kernelM[i, j] = ot.emd2(u, u, M)  # the wasserstein distance for two of the graphs



    kernelM = np.exp(-10*kernelM)
    return kernelM



def compute_euclidian(list1, list2): #this calculates the euclidian distance instead of the Wasserstein distance. Much quicker to compute.

    kernelM = np.empty((len(list2), len(list1)))
    print(kernelM.shape)# empty kernel matrix

    for i in range(len(list2)):
        if i % 200 == 0:
            print('Euclidian:' + str(i + 1) + '/' + str(len(list2)))

        for j in range(len(list1)):

            kernelM[i, j] = np.linalg.norm(list2[i] - list1[j])

    kernelM = np.exp(-kernelM)
    return kernelM


#the following function allows to either compute the WWL or EWL kernel (using the label euclidean=True). For practical purposes the train and test kernels are always computed together.

def compute_WWL_ttsplit(A_list, A_list_test,node_label_list, node_label_list_test,depth, euclidian=False):#A_list is a list of adjacency matrices, node_label_list is the list of node labels, and depth is the same as before, ld a positive parameter

    node_label_list = assimilate_labels(A_list, node_label_list, depth)
    node_label_list_test = assimilate_labels(A_list_test, node_label_list_test, depth)

    if euclidian:
        test_kernel = compute_euclidian(node_label_list, node_label_list_test)
        train_kernel = compute_euclidian(node_label_list, node_label_list)
    else:
        test_kernel = compute_wasserstein(node_label_list, node_label_list_test)
        train_kernel = compute_wasserstein(node_label_list, node_label_list)


    return train_kernel, test_kernel







