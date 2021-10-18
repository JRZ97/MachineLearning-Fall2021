"""
This is the script that will run through HW2 Problem 4. 
"""

import numpy as np
import csv
import LMS # import LMS functions

# =============================================================================
# Problem 4
# =============================================================================
############################## load in the data ###############################
with open('concrete/train.csv', newline='') as f:
    reader = csv.reader(f)
    P4_train = np.transpose(np.array(list(reader)).astype('float64'))
    # feature augmentation
    train_augmentArr = np.ones(np.size(P4_train,1))
    P4_trainAugment = np.vstack((train_augmentArr, P4_train))
    
with open('concrete/test.csv', newline='') as f:
    reader = csv.reader(f)
    P4_test = np.transpose(np.array(list(reader)).astype('float64'))
    # feature augmentation
    test_augmentArr = np.ones(np.size(P4_test,1))
    P4_testAugment = np.vstack((test_augmentArr, P4_test))

################################## Part (a) ###################################
r = 0.015 # learning rate
thresh = 1e-6 # error threshold for norm of weight vector difference

w_BGD, J_BGDtrain = LMS.BGD(P4_trainAugment[:-1,:], P4_trainAugment[-1,:], r, thresh)

# calculate cost value function over test data
X = P4_testAugment[:-1,:]
y = P4_testAugment[-1,:]
err_arr = np.array([y[i] - np.matmul(w_BGD, X[:,i]) for i in range(np.shape(P4_testAugment)[1])])
J_BGDtest = 0.5 * sum(np.square(err_arr))

# # plot the cost function value
# import matplotlib.pyplot as plt
# plt.plot(np.arange(np.size(J_BGDtrain)), J_BGDtrain)
# plt.title('Cost function value vs. steps for batch gradient descent')
# plt.xlabel('Steps')
# plt.ylabel('Cost function value')
# plt.show()

# print out the results
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print("************** PROBLEM 4 PART (a) **************")
print(f"For batch gradient descent, final weight vector is {w_BGD}, the learning rate is {r}, and the cost function value over the test data is {J_BGDtest:0.4f}.")
print("")

################################## Part (b) ###################################
r = 0.003 # learning rate
thresh = 1e-6 # error threshold for norm of cost evaluation difference

w_SGD, J_SGDtrain = LMS.SGD(P4_trainAugment[:-1,:], P4_trainAugment[-1,:], r, thresh)

# calculate cost value function over test data
X = P4_testAugment[:-1,:]
y = P4_testAugment[-1,:]
err_arr = np.array([y[i] - np.matmul(w_SGD, X[:,i]) for i in range(np.shape(P4_testAugment)[1])])
J_SGDtest = 0.5 * sum(np.square(err_arr))

# # plot the cost function value
# import matplotlib.pyplot as plt
# plt.plot(np.arange(np.size(J_SGDtrain)), J_SGDtrain)
# plt.title('Cost function value vs. steps for stochastic gradient descent')
# plt.xlabel('Steps')
# plt.ylabel('Cost function value')
# plt.show()

# print out the results
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print("************** PROBLEM 4 PART (b) **************")
print(f"For batch gradient descent, final weight vector is {w_SGD}, the learning rate is {r}, and the cost function value over the test data is {J_SGDtest:0.4f}.")
print("")

################################## Part (c) ###################################
w_analyt = LMS.analyt(P4_trainAugment[:-1,:], P4_trainAugment[-1,:])
# print out the results
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print("************** PROBLEM 4 PART (c) **************")
print(f"The analytical weight vector is {w_analyt}.")