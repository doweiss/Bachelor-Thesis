import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score

savepath = 'path to save images to'


labels_test_list = np.load() #load the train and test labels
labels_list = np.load()

#load the precomputed kernels
kernelTrain = np.load('path to kernel matrix.npy', allow_pickle=True)
kernelTest = np.load('path to kernel matrix.npy', allow_pickle=True)

#define and fit SVM, if C needs to be found use this in a for loop to do grid search for C
clf = svm.SVC(kernel='precomputed', C=1)
clf.fit(kernelTrain, labels_list)

#print f1 score, bal acc and confusion matrix for test set
prediction = clf.predict(kernelTest)

print('f1_score:')
print(f1_score(labels_test_list, prediction))

print('balanced accuracy:')
print(balanced_accuracy_score(labels_test_list, prediction))

C = confusion_matrix(labels_test_list, prediction)
print(C)

#plot histogram of projections for training kernel, do the same thing for test kernel if wanted
decfun = clf.decision_function(kernelTrain)#distance to hyperplane

fig = plt.figure()
ax = fig.add_subplot()

labels_list = np.array(labels_list)
indstrue = np.where(labels_list == 1)
indsfalse = np.where(labels_list == 0)
true = decfun[indstrue]
false = decfun[indsfalse]


plt.hist(true, bins=100, color='blue', label='Cascade')
plt.hist(false, bins=100, color='orange', label='No Cascade')
plt.legend(loc="upper right")
plt.xlabel('Distance from the decision boundary')
plt.ylabel('Number of samples')
plt.savefig(savepath + "filename")
plt.show()



