import numpy as np
import csv
import matplotlib.pyplot as plt
from logistic import LOGSTC

# use when diagnosing convergence
# plt.plot(np.arange(epochs*np.size(X_train,0)), J)
# plt.title('Squared loss vs. number of updates')
# plt.xlabel('Number of updates')
# plt.ylabel('Objective function value')
# plt.show()

####################### load in the data #######################
with open('bank-note/train.csv', newline='') as f:
    reader = csv.reader(f)
    HW5_train = np.array(list(reader)).astype('float32')
    
with open('bank-note/test.csv', newline='') as f:
    reader = csv.reader(f)
    HW5_test = np.array(list(reader)).astype('float32')

y_train = np.array([1 if label==1 else -1 for label in HW5_train[:,-1]])
y_test = np.array([1 if label==1 else -1 for label in HW5_test[:,-1]])

X_train = np.hstack((np.ones(np.size(HW5_train,0))[:, np.newaxis], HW5_train[:,:-1]))
X_test = np.hstack((np.ones(np.size(HW5_test,0))[:, np.newaxis], HW5_test[:,:-1]))

# ============================================================================
# Problem 3
# ============================================================================

############################### Part (a) #####################################
gamma_0Arr = [5e-1, 1e-1, 1e-1, 5e-2, 5e-2, 5e-2, 5e-2, 1e-2]
a_arr = [1e-2, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
var_arr = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
epochs = 100
train_type = "MAP"
P3_a = LOGSTC(X_train,y_train)

for i in range(len(var_arr)):
    v = var_arr[i]
    P3_a.sgd(epochs, gamma_0Arr[i], a_arr[i], v, train_type, conv_test = "F")
    
    # find train/test errors
    y_trainPred = P3_a.pred(X_train)
    trainErrArr = np.multiply(y_train,y_trainPred)
    numTrainErr = (trainErrArr == -1).sum()
    trainErr = numTrainErr/np.size(y_trainPred)
    print(f"For Problem 3(a), variance = {v}, training error is {trainErr:0.4f}")
    
    y_testPred = P3_a.pred(X_test)
    testErrArr = np.multiply(y_test,y_testPred)
    numTestErr = (testErrArr == -1).sum()
    testErr = numTestErr/np.size(y_testPred)
    print(f"For Problem 3(a), variance = {v}, test error is {testErr:0.4f}")

############################### Part (b) #####################################
gamma_0 = 1e-2
a = 1e-3
v = 0.01 # doesn't actually matter
epochs = 100
train_type = "MLE"
P3_b = LOGSTC(X_train,y_train)

P3_b.sgd(epochs, gamma_0, a, v, train_type, conv_test = "F")

# find train/test errors
y_trainPred = P3_b.pred(X_train)
trainErrArr = np.multiply(y_train,y_trainPred)
numTrainErr = (trainErrArr == -1).sum()
trainErr = numTrainErr/np.size(y_trainPred)
print(f"For Problem 3(b), training error is {trainErr:0.4f}")

y_testPred = P3_b.pred(X_test)
testErrArr = np.multiply(y_test,y_testPred)
numTestErr = (testErrArr == -1).sum()
testErr = numTestErr/np.size(y_testPred)
print(f"For Problem 3(b), test error is {testErr:0.4f}")