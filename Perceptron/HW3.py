import numpy as np
import csv
from perceptron import PCPT

# ============================================================================
# Problem 2
# ============================================================================
############################## load in the data ##############################
with open('bank-note/train.csv', newline='') as f:
    reader = csv.reader(f)
    P2_train = np.array(list(reader)).astype('float32')
    
with open('bank-note/test.csv', newline='') as f:
    reader = csv.reader(f)
    P2_test = np.array(list(reader)).astype('float32')

y_train = np.array([1 if label==1 else -1 for label in P2_train[:,-1]])
y_test = np.array([1 if label==1 else -1 for label in P2_test[:,-1]])

X_train = np.hstack((np.ones(np.size(P2_train,0))[:, np.newaxis],P2_train[:,:-1]))
X_test = np.hstack((np.ones(np.size(P2_test,0))[:, np.newaxis],P2_test[:,:-1]))

P2_PCPT = PCPT(X_train,y_train,epochs=10,r=0.01)

# standard algorithm
P2_PCPT.std_alg()
y_stdPred = P2_PCPT.std_pred(X_test)
std_PredErrArr = np.multiply(y_test,y_stdPred)
std_numPredErr = (std_PredErrArr == -1).sum()
std_PredErr = std_numPredErr/np.size(y_stdPred)
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"The weight vector for the standard algorithm is {P2_PCPT.w_std}.")
    print(f"The average prediction error for the standard algorithm on the test dataset is {std_PredErr}.")

# voted algorithm
P2_PCPT.vot_alg()
y_votPred = P2_PCPT.vot_pred(X_test)
vot_PredErrArr = np.multiply(y_test,y_votPred)
vot_numPredErr = (vot_PredErrArr == -1).sum()
vot_PredErr = vot_numPredErr/np.size(y_votPred)
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"Some of the weight vectors for the voted algorithm are {P2_PCPT.w_vot[0]}, {P2_PCPT.w_vot[1]}, ..., {P2_PCPT.w_vot[-1]}.")
    print(f"The counts for the weight vectors for the voted algorithm are {P2_PCPT.c_vot}.")
    print(f"The average prediction error for the voted algorithm on the test dataset is {vot_PredErr}.")

# averaged algorithm
P2_PCPT.avg_alg()
y_avgPred = P2_PCPT.avg_pred(X_test)
avg_PredErrArr = np.multiply(y_test,y_avgPred)
avg_numPredErr = (avg_PredErrArr == -1).sum()
avg_PredErr = avg_numPredErr/np.size(y_avgPred)
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"The weight vector for the averaged algorithm is {P2_PCPT.a_avg}.")
    print(f"The average prediction error for the standard algorithm on the test dataset is {avg_PredErr}.")




