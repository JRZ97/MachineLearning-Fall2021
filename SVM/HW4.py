import numpy as np
import csv
from SVM import SVM
import matplotlib.pyplot as plt

####################### load in the data #######################
with open('bank-note/train.csv', newline='') as f:
    reader = csv.reader(f)
    P2_train = np.array(list(reader)).astype('float32')
    
with open('bank-note/test.csv', newline='') as f:
    reader = csv.reader(f)
    P2_test = np.array(list(reader)).astype('float32')

y_train = np.array([1 if label==1 else -1 for label in P2_train[:,-1]])
y_test = np.array([1 if label==1 else -1 for label in P2_test[:,-1]])

X_train = np.hstack((P2_train[:,:-1], np.ones(np.size(P2_train,0))[:, np.newaxis]))
X_test = np.hstack((P2_test[:,:-1], np.ones(np.size(P2_test,0))[:, np.newaxis]))

# ============================================================================
# Problem 2
# ============================================================================
P2 = SVM(X_train, y_train) # instantiate SVM class

####################### train primal form of SVM #######################
epochs = 100
C_arr = [100/873, 500/873, 700/873]

######### Part (a) #########
### C = 100/873 ###
w_C0_a = P2.prim_alg(epochs, C_arr[0], eval_J = 0, gamma_sched = '2a', gamma_0 = 1e-3, a = 1e-4)

# use when diagnosing convergence
# w_C0_a, J = P2.prim_alg(epochs, C_arr[0], gamma_sched = '2a', gamma_0 = 1e-3, a = 1e-4)
# plt.plot(np.arange(epochs*np.size(X_train,0)), J)
# plt.title('Objective function value vs. number of updates for Probem 2(a), C = 100/873')
# plt.xlabel('Number of updates')
# plt.ylabel('Objective function value')
# plt.show()

# find train/test errors
y_primTrainPred = P2.prim_pred(X_train)
prim_trainErrArr = np.multiply(y_train,y_primTrainPred)
prim_numTrainErr = (prim_trainErrArr == -1).sum()
prim_trainErr = prim_numTrainErr/np.size(y_primTrainPred)

y_primTestPred = P2.prim_pred(X_test)
prim_testErrArr = np.multiply(y_test,y_primTestPred)
prim_numTestErr = (prim_testErrArr == -1).sum()
prim_testErr = prim_numTestErr/np.size(y_primTestPred)

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"For Problem 2(a), C = 100/873:")
    print(f"\t The weight vector is {w_C0_a}.")
    print(f"\t The training error is {prim_trainErr:0.4f}.")
    print(f"\t The test error is {prim_testErr:0.4f}.")
    print("")

### C = 500/873 ###
w_C1_a = P2.prim_alg(epochs, C_arr[1], eval_J = 0, gamma_sched = '2a', gamma_0 = 5e-4, a = 1e-5)

# use when diagnosing convergence
# w_C1_a, J = P2.prim_alg(epochs, C_arr[1], gamma_sched = '2a', gamma_0 = 5e-4, a = 1e-5)
# plt.plot(np.arange(epochs*np.size(X_train,0)), J)
# plt.title('Objective function value vs. number of updates for Probem 2(a), C = 500/873')
# plt.xlabel('Number of updates')
# plt.ylabel('Objective function value')
# plt.show()

# find train/test errors
y_primTrainPred = P2.prim_pred(X_train)
prim_trainErrArr = np.multiply(y_train,y_primTrainPred)
prim_numTrainErr = (prim_trainErrArr == -1).sum()
prim_trainErr = prim_numTrainErr/np.size(y_primTrainPred)

y_primTestPred = P2.prim_pred(X_test)
prim_testErrArr = np.multiply(y_test,y_primTestPred)
prim_numTestErr = (prim_testErrArr == -1).sum()
prim_testErr = prim_numTestErr/np.size(y_primTestPred)

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"For Problem 2(a), C = 500/873:")
    print(f"\t The weight vector is {w_C1_a}.")
    print(f"\t The training error is {prim_trainErr:0.4f}.")
    print(f"\t The test error is {prim_testErr:0.4f}.")
    print("")

### C = 700/873 ###
w_C2_a = P2.prim_alg(epochs, C_arr[2], eval_J = 0, gamma_sched = '2a', gamma_0 = 5e-4, a = 1e-5)

# use when diagnosing convergence
# w_C2_a, J = P2.prim_alg(epochs, C_arr[2], gamma_sched = '2a', gamma_0 = 5e-4, a = 1e-5)
# plt.plot(np.arange(epochs*np.size(X_train,0)), J)
# plt.title('Objective function value vs. number of updates for Probem 2(a), C = 700/873')
# plt.xlabel('Number of updates')
# plt.ylabel('Objective function value')
# plt.show()

# find train/test errors
y_primTrainPred = P2.prim_pred(X_train)
prim_trainErrArr = np.multiply(y_train,y_primTrainPred)
prim_numTrainErr = (prim_trainErrArr == -1).sum()
prim_trainErr = prim_numTrainErr/np.size(y_primTrainPred)

y_primTestPred = P2.prim_pred(X_test)
prim_testErrArr = np.multiply(y_test,y_primTestPred)
prim_numTestErr = (prim_testErrArr == -1).sum()
prim_testErr = prim_numTestErr/np.size(y_primTestPred)

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"For Problem 2(a), C = 700/873:")
    print(f"\t The weight vector is {w_C2_a}.")
    print(f"\t The training error is {prim_trainErr:0.4f}.")
    print(f"\t The test error is {prim_testErr:0.4f}.")
    print("")

######### Part (b) #########
### C = 100/873 ###
w_C0_b = P2.prim_alg(epochs, C_arr[0], eval_J = 0, gamma_sched = '2b', gamma_0 = 2e-4)

# use when diagnosing convergence
# epochs = 20
# w_C0_b, J = P2.prim_alg(epochs, C_arr[0], gamma_sched = '2b', gamma_0 = 2e-4)
# plt.plot(np.arange(epochs*np.size(X_train,0)), J)
# plt.title('Objective function value vs. number of updates for Probem 2(b), C = 100/873')
# plt.xlabel('Number of updates')
# plt.ylabel('Objective function value')
# plt.show()

# find train/test errors
y_primTrainPred = P2.prim_pred(X_train)
prim_trainErrArr = np.multiply(y_train,y_primTrainPred)
prim_numTrainErr = (prim_trainErrArr == -1).sum()
prim_trainErr = prim_numTrainErr/np.size(y_primTrainPred)

y_primTestPred = P2.prim_pred(X_test)
prim_testErrArr = np.multiply(y_test,y_primTestPred)
prim_numTestErr = (prim_testErrArr == -1).sum()
prim_testErr = prim_numTestErr/np.size(y_primTestPred)

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"For Problem 2(b), C = 100/873:")
    print(f"\t The weight vector is {w_C0_b}.")
    print(f"\t The training error is {prim_trainErr:0.4f}.")
    print(f"\t The test error is {prim_testErr:0.4f}.")
    print("")

### C = 500/873 ###
w_C1_b = P2.prim_alg(epochs, C_arr[1], eval_J = 0, gamma_sched = '2b', gamma_0 = 3e-5)

# use when diagnosing convergence
# epochs = 20
# w_C1_b, J = P2.prim_alg(epochs, C_arr[1], gamma_sched = '2b', gamma_0 = 3e-5)
# plt.plot(np.arange(epochs*np.size(X_train,0)), J)
# plt.title('Objective function value vs. number of updates for Probem 2(b), C = 500/873')
# plt.xlabel('Number of updates')
# plt.ylabel('Objective function value')
# plt.show()

# find train/test errors
y_primTrainPred = P2.prim_pred(X_train)
prim_trainErrArr = np.multiply(y_train,y_primTrainPred)
prim_numTrainErr = (prim_trainErrArr == -1).sum()
prim_trainErr = prim_numTrainErr/np.size(y_primTrainPred)

y_primTestPred = P2.prim_pred(X_test)
prim_testErrArr = np.multiply(y_test,y_primTestPred)
prim_numTestErr = (prim_testErrArr == -1).sum()
prim_testErr = prim_numTestErr/np.size(y_primTestPred)

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"For Problem 2(b), C = 500/873:")
    print(f"\t The weight vector is {w_C1_b}.")
    print(f"\t The training error is {prim_trainErr:0.4f}.")
    print(f"\t The test error is {prim_testErr:0.4f}.")
    print("")

### C = 700/873 ###
w_C2_b = P2.prim_alg(epochs, C_arr[2], eval_J = 0, gamma_sched = '2b', gamma_0 = 2e-5)

# use when diagnosing convergence
# epochs = 20
# w_C2_b, J = P2.prim_alg(epochs, C_arr[2], gamma_sched = '2b', gamma_0 = 2e-5)
# plt.plot(np.arange(epochs*np.size(X_train,0)), J)
# plt.title('Objective function value vs. number of updates for Probem 2(b), C = 700/873')
# plt.xlabel('Number of updates')
# plt.ylabel('Objective function value')
# plt.show()

# find train/test errors
y_primTrainPred = P2.prim_pred(X_train)
prim_trainErrArr = np.multiply(y_train,y_primTrainPred)
prim_numTrainErr = (prim_trainErrArr == -1).sum()
prim_trainErr = prim_numTrainErr/np.size(y_primTrainPred)

y_primTestPred = P2.prim_pred(X_test)
prim_testErrArr = np.multiply(y_test,y_primTestPred)
prim_numTestErr = (prim_testErrArr == -1).sum()
prim_testErr = prim_numTestErr/np.size(y_primTestPred)

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"For Problem 2(b), C = 700/873:")
    print(f"\t The weight vector is {w_C2_b}.")
    print(f"\t The training error is {prim_trainErr:0.4f}.")
    print(f"\t The test error is {prim_testErr:0.4f}.")
    print("")

# ============================================================================
# Problem 3
# ============================================================================
P3 = SVM(X_train, y_train) # instantiate SVM class

####################### train dual form of SVM #######################
C_arr = [100/873, 500/873, 700/873]

######### Part (a) #########
### C = 100/873 ###
w_C0 = P3.dual_alg(C_arr[0])

# find train/test errors
y_dualTrainPred = P3.dual_pred(X_train)
dual_trainErrArr = np.multiply(y_train,y_dualTrainPred)
dual_numTrainErr = (dual_trainErrArr == -1).sum()
dual_trainErr = dual_numTrainErr/np.size(y_dualTrainPred)

y_dualTestPred = P3.dual_pred(X_test)
dual_testErrArr = np.multiply(y_test,y_dualTestPred)
dual_numTestErr = (dual_testErrArr == -1).sum()
dual_testErr = dual_numTestErr/np.size(y_dualTestPred)

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"For Problem 3(a), C = 100/873:")
    print(f"\t The weight vector is {w_C0}.")
    print(f"\t The training error is {dual_trainErr:0.4f}.")
    print(f"\t The test error is {dual_testErr:0.4f}.")
    print("")

### C = 500/873 ###
w_C1 = P3.dual_alg(C_arr[1])

# find train/test errors
y_dualTrainPred = P3.dual_pred(X_train)
dual_trainErrArr = np.multiply(y_train,y_dualTrainPred)
dual_numTrainErr = (dual_trainErrArr == -1).sum()
dual_trainErr = dual_numTrainErr/np.size(y_dualTrainPred)

y_dualTestPred = P3.dual_pred(X_test)
dual_testErrArr = np.multiply(y_test,y_dualTestPred)
dual_numTestErr = (dual_testErrArr == -1).sum()
dual_testErr = dual_numTestErr/np.size(y_dualTestPred)

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"For Problem 3(a), C = 500/873:")
    print(f"\t The weight vector is {w_C1}.")
    print(f"\t The training error is {dual_trainErr:0.4f}.")
    print(f"\t The test error is {dual_testErr:0.4f}.")
    print("")

### C = 700/873 ###
w_C2 = P3.dual_alg(C_arr[2])

# find train/test errors
y_dualTrainPred = P3.dual_pred(X_train)
dual_trainErrArr = np.multiply(y_train,y_dualTrainPred)
dual_numTrainErr = (dual_trainErrArr == -1).sum()
dual_trainErr = dual_numTrainErr/np.size(y_dualTrainPred)

y_dualTestPred = P3.dual_pred(X_test)
dual_testErrArr = np.multiply(y_test,y_dualTestPred)
dual_numTestErr = (dual_testErrArr == -1).sum()
dual_testErr = dual_numTestErr/np.size(y_dualTestPred)

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(f"For Problem 3(a), C = 700/873:")
    print(f"\t The weight vector is {w_C2}.")
    print(f"\t The training error is {dual_trainErr:0.4f}.")
    print(f"\t The test error is {dual_testErr:0.4f}.")
    print("")

######### Parts (b) and (c) #########
gamma_arr = [0.1, 0.5, 1, 5, 100]
thresh = 1e-6

### C = 100/873 ###
C = C_arr[0]
for gamma in gamma_arr:
    P3.dual_NLalg(C, kern = 'Gauss', gamma = gamma)
    
    # find train/test errors
    y_dualTrainPred = P3.dual_NLpred(X_train)
    dual_trainErrArr = np.multiply(y_train,y_dualTrainPred)
    dual_numTrainErr = (dual_trainErrArr == -1).sum()
    dual_trainErr = dual_numTrainErr/np.size(y_dualTrainPred)
    
    y_dualTestPred = P3.dual_NLpred(X_test)
    dual_testErrArr = np.multiply(y_test,y_dualTestPred)
    dual_numTestErr = (dual_testErrArr == -1).sum()
    dual_testErr = dual_numTestErr/np.size(y_dualTestPred)
    
    numSupV = sum([1 if alpha > thresh else 0 for alpha in P3.NL_alphaStar])
    
    with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
        print(f"For Problems 3(b) and 3(c), C = 100/873, gamma = {gamma}:")
        print(f"\t The training error is {dual_trainErr:0.4f}.")
        print(f"\t The test error is {dual_testErr:0.4f}.")
        print(f"\t The number of support vectors is {numSupV}.")
        print("")

### C = 500/873 ###
C = C_arr[1]
supV_arr = []
alpha_arr = []
for gamma in gamma_arr:
    alpha_arr.append(P3.dual_NLalg(C, kern = 'Gauss', gamma = gamma))
    
    # find train/test errors
    y_dualTrainPred = P3.dual_NLpred(X_train)
    dual_trainErrArr = np.multiply(y_train,y_dualTrainPred)
    dual_numTrainErr = (dual_trainErrArr == -1).sum()
    dual_trainErr = dual_numTrainErr/np.size(y_dualTrainPred)
    
    y_dualTestPred = P3.dual_NLpred(X_test)
    dual_testErrArr = np.multiply(y_test,y_dualTestPred)
    dual_numTestErr = (dual_testErrArr == -1).sum()
    dual_testErr = dual_numTestErr/np.size(y_dualTestPred)
    
    supV = np.array([1 if alpha > thresh else 0 for alpha in P3.NL_alphaStar])
    supV_arr.append(supV)
    numSupV = sum(supV)
    
    with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
        print(f"For Problem 3(b) and 3(c), C = 500/873, gamma = {gamma}:")
        print(f"\t The training error is {dual_trainErr:0.4f}.")
        print(f"\t The test error is {dual_testErr:0.4f}.")
        print(f"\t The number of support vectors is {numSupV}.")
        print("")

# list overlapping support vectors between consecutive gammas
g01g05_numOverlap = sum(supV_arr[0] * supV_arr[1])
print(f"\t There are {g01g05_numOverlap} overlapped support vectors between gamma = 0.1 and gamma = 0.5.")
g05g1_numOverlap = sum(supV_arr[1] * supV_arr[2])
print(f"\t There are {g05g1_numOverlap} overlapped support vectors between gamma = 0.5 and gamma = 1.")
g1g5_numOverlap = sum(supV_arr[2] * supV_arr[3])
print(f"\t There are {g1g5_numOverlap} overlapped support vectors between gamma = 1 and gamma = 5.")
g5g100_numOverlap = sum(supV_arr[3] * supV_arr[4])
print(f"\t There are {g5g100_numOverlap} overlapped support vectors between gamma = 5 and gamma = 100.")
print("")
      
### C = 700/873 ###
C = C_arr[2]
for gamma in gamma_arr:
    P3.dual_NLalg(C, kern = 'Gauss', gamma = gamma)
    
    # find train/test errors
    y_dualTrainPred = P3.dual_NLpred(X_train)
    dual_trainErrArr = np.multiply(y_train,y_dualTrainPred)
    dual_numTrainErr = (dual_trainErrArr == -1).sum()
    dual_trainErr = dual_numTrainErr/np.size(y_dualTrainPred)
    
    y_dualTestPred = P3.dual_NLpred(X_test)
    dual_testErrArr = np.multiply(y_test,y_dualTestPred)
    dual_numTestErr = (dual_testErrArr == -1).sum()
    dual_testErr = dual_numTestErr/np.size(y_dualTestPred)
    
    numSupV = sum([1 if alpha > thresh else 0 for alpha in P3.NL_alphaStar])
    
    with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
        print(f"For Problems 3(b) and 3(c), C = 700/873, gamma = {gamma}:")
        print(f"\t The training error is {dual_trainErr:0.4f}.")
        print(f"\t The test error is {dual_testErr:0.4f}.")
        print(f"\t The number of support vectors is {numSupV}.")




















