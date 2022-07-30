import numpy as np

#this function computes the Weisfeiler Lehman embeddings or counter vectors for a given list of adjacency matrices

def compute_WL_embeddings(A_list, depth): #A_list: A_list is a python list of numpy adjacency matrices, the kernel is computed for depth-hop neighborhood of each node
    n = len(A_list)

    #make an empty list for the new labels and the counter (the vector in which we count the number a color/label occurs in a given graph)
    labels_list = [[] for x in range(n)]
    counters_list = [[] for x in range(n)]

    #a list of all existing labels, in this case where all nodes are the same, all of them just have a label of 1, so this is already added
    all_labels = np.array([1])

    #a list of all the neighborhood vectors that belong to the labels (this and all_labels have to strictly have the same order)
    hashes = []

    #if their are different starting labels add all of the to the list of neighborhood vectors
    for label in all_labels:
        hashes.append(np.array([label]).astype(int))

    label_number = 0

    #for every adjacency matrix in the list
    for A in tqdm(A_list):
        #a counter

        label_number = label_number + 1

        labels = np.empty(0)
        for i in range(A.shape[0]):
            labels = np.append(labels, 1)

        #do the entire thing depth times (normally around 3, which means that for every node its 3hop neighborhood is considered)
        for d in range(depth):
            #i update the label vector, but i have to work with the old labels
            old_labels = np.copy(labels)

            #iterate through all the nodes in a graph
            for i in range(labels.shape[0]):

                #this is the new label/list of adjacent labels for the current node
                new_label = np.empty(0)

                #for all the nodes adjacent to the current node, retrieve their labels and put it into a list
                for j in range(A.shape[1]):
                    if A[i][j] == 1 and i != j:
                        new_label = np.append(new_label, old_labels[j])

                #sort the list of adjacent labels (this is so different permutations are treated the same later on)
                new_label = np.sort(new_label)
                #prepend the old labels to the list of labels of adjacent nodes (as is done in the paper, once again to prevent nodes that have different labels but the same neighborhood from being treated the same)
                new_label = np.append(np.array([old_labels[i]]), new_label)

                #this now assigns a new label to the node which is unique to a specific new_label vector
                bool = False
                for it in range(all_labels.shape[0]):
                    if np.array_equal(hashes[it], new_label.astype(int)): #if a given new_label has already occured give it the same label as before
                        new_label = all_labels[it]
                        bool = True

                #if this neighborhood has not occured yet a new label for it is created (which is just one higher than the current highest)
                if bool != True:
                    hashes.append(new_label.astype(int))
                    new_label = np.max(all_labels) + 1
                    all_labels = np.append(all_labels, np.max(all_labels) + 1)

                #if new_label is empty, this means the node is an island and just keeps its label
                elif (new_label.size == 0):
                    new_label = old_labels[i]

                #the label of the current node is updatet to the new label (however for all other nodes in this iteration we still work with the old nodes)
                labels[i] = new_label

        #after all is finished for a given A matrix, its new labels are added to the list of labels
        labels_list[label_number-1] = labels

    #count number of occurences for every color in the new node label vector of each graph
    counters_number = 0
    #for every graph
    for labels in labels_list:
        counter = np.empty(0)
        counters_number = counters_number + 1
        #and for every label that might occur in the graph (all labels carries all the possible labels)
        for a in all_labels:
            counter = np.append(counter, np.count_nonzero(labels==a))
        counters_list[counters_number-1] = counter

    #the kernel matrix then is simply the dot product of all the counter lists with themselves

    return counters_list

#this function computes the WL kernel for training and test set. This has to be done simultaneously as explained below!
def compute_WL_ttsplit(A_train_list, A_test_list, depth):
    n = len(A_train_list)#find the length of the training list
    for A in A_test_list: #append one list to the other
        A_train_list.append(A)
    embeddings = compute_WL_embeddings(A_train_list, depth) #this is done as the embeddings have to be computed for the test and training set together
    #this limitation can be solved by using a hash function. This was not needed in this thesis and not using a hash makes the function simpler.

    #split the embeddings back up
    embeddings_train = embeddings[:n]
    embeddings_test = embeddings[n:]

    #compute the dot products
    kernelTrain = np.dot(np.array(embeddings_train), np.array(embeddings_train).T)
    kernelTest = np.dot(np.array(embeddings_train), np.array(embeddings_test).T)

    return kernelTrain, kernelTest











