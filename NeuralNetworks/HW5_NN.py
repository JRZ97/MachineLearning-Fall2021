import numpy as np
import csv
from NN import NN
import matplotlib.pyplot as plt

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
# Problem 2
# ============================================================================

############################### Part (a) #####################################
X = np.array([[1,1,1]])
y = 1 
d = 3

print(f"Problem 2(a)")
print(f"To verify forward and back propagation, I repeated Paper Problems 2 and 3 in the code:")
print("")
P2_a = NN(X,y,d,"test") # instantiate NN class
P2_a._fwd_prop(X[0],y_star=y)
P2_a._back_prop(X[0])

############################### Part (b) #####################################
epochs = 50
width_arr = [5, 10, 25, 50, 100]
gamma_0Arr = [7e-1, 7e-1, 2e-1, 1e-1, 5e-2]
a_arr = [7e-1, 7e-1, 2e-1, 1e-1, 5e-2]

for i in range(len(width_arr)):
    P2_b = NN(X_train,y_train,width_arr[i],"normal")
    P2_b.sgd(epochs, gamma_0Arr[i], a_arr[i])
    J = P2_b.loss
    
    # find train/test errors
    y_trainPred = P2_b.pred(X_train)
    trainErrArr = np.multiply(y_train,y_trainPred)
    numTrainErr = (trainErrArr == -1).sum()
    trainErr = numTrainErr/np.size(y_trainPred)
    print(f"For Problem 2(b), width = {width_arr[i]}, training error is {trainErr:0.4f}")
    
    y_testPred = P2_b.pred(X_test)
    testErrArr = np.multiply(y_test,y_testPred)
    numTestErr = (testErrArr == -1).sum()
    testErr = numTestErr/np.size(y_testPred)
    print(f"For Problem 2(b), width = {width_arr[i]}, test error is {testErr:0.4f}")
    
############################### Part (c) #####################################
for i in range(len(width_arr)):
    P2_b = NN(X_train,y_train,width_arr[i],"zeros")
    P2_b.sgd(epochs, gamma_0Arr[i], a_arr[i])
    J = P2_b.loss
    
    # find train/test errors
    y_trainPred = P2_b.pred(X_train)
    trainErrArr = np.multiply(y_train,y_trainPred)
    numTrainErr = (trainErrArr == -1).sum()
    trainErr = numTrainErr/np.size(y_trainPred)
    print(f"For Problem 2(c), width = {width_arr[i]}, training error is {trainErr:0.4f}")
    
    y_testPred = P2_b.pred(X_test)
    testErrArr = np.multiply(y_test,y_testPred)
    numTestErr = (testErrArr == -1).sum()
    testErr = numTestErr/np.size(y_testPred)
    print(f"For Problem 2(c), width = {width_arr[i]}, test error is {testErr:0.4f}")
