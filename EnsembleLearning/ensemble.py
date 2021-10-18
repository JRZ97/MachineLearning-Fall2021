"""
This is the script that will run through HW2 Problem 2. 
"""

import numpy as np
import pandas as pd
import csv
from decision_tree import DT # import DT class
from AdaBoost import AdaBoost # import AdaBoost class

# =============================================================================
# Problem 2
# =============================================================================
############################## load in the data ###############################
# go through and convert numeric features to binary, leaving unknown as an 
# attribute value. 
with open('bank-1/train.csv', newline='') as f:
    reader = csv.reader(f)
    P2_train_prelimData = np.array(list(reader))
    
with open('bank-1/test.csv', newline='') as f:
    reader = csv.reader(f)
    P2_test_prelimData = np.array(list(reader))

P2_trainData = P2_train_prelimData.copy() # modified categorical array
P2_testData = P2_test_prelimData.copy() # modified categorical array

categ_columns = [0,5,9,11,12,13,14]
for column in categ_columns:
    columnTrain_prelim = P2_train_prelimData[:,column].astype(float)
    columnTest_prelim = P2_test_prelimData[:,column].astype(float)
    med_valTrain = np.median(columnTrain_prelim)

    columnTrain_categ = []
    for x in columnTrain_prelim:
        if x >= med_valTrain:
            columnTrain_categ.append('High')
        else:
            columnTrain_categ.append('Low')
    P2_trainData[:,column] = columnTrain_categ

    columnTest_categ = []
    for x in columnTest_prelim:
        if x >= med_valTrain:
            columnTest_categ.append('High')
        else:
            columnTest_categ.append('Low')
    P2_testData[:,column] = columnTest_categ

P2_trainData[:,-1] = np.array([1 if label_prelim=='yes' else -1 for label_prelim in P2_train_prelimData[:,-1]])
P2_testData[:,-1] = np.array([1 if label_prelim=='yes' else -1 for label_prelim in P2_test_prelimData[:,-1]])

# read in data dictionary with possible values
data_desc = {
    'age': ['High','Low'],
    'job': ['admin.','unknown','unemployed','management','housemaid',
            'entrepreneur','student','blue-collar','self-employed',
            'retired','technician','services'],
    'marital': ['married','divorced','single'],
    'education': ['unknown','secondary','primary','tertiary'],
    'default': ['yes','no'],
    'balance': ['High','Low'],
    'housing': ['yes','no'],
    'loan': ['yes','no'],
    'contact': ['unknown','telephone','cellular'],
    'day': ['High','Low'],
    'month': ['jan','feb','mar','apr','may','jun','jul','aug',
              'sep','oct','nov','dec'],
    'duration': ['High','Low'],
    'campaign': ['High','Low'],
    'pdays': ['High','Low'],
    'previous': ['High','Low'],
    'poutcome': ['unknown','other','failure','success'],
    'y': [-1, 1]}

trainData_df = pd.DataFrame(P2_trainData,columns=data_desc.keys())
testData_df = pd.DataFrame(P2_testData,columns=data_desc.keys())

# separate target from predictors
X_train = np.array(trainData_df.drop('y', axis=1).copy())
y_train = np.array(trainData_df['y'].copy())
y_train = y_train.astype('int64')

X_test= np.array(testData_df.drop('y', axis=1).copy())
y_test = np.array(testData_df['y'].copy())
y_test = y_test.astype('int64')

# number of training and test examples
num_trainExamples = np.size(X_train,0)
num_testExamples = np.size(X_test,0)

# make dictionaries representing the train/test X for predictions
X_trainDict = [] # list of dictionaries
for row  in X_train:
    example_dict = {'age': row[0],
                    'job': row[1],
                    'marital': row[2],
                    'education': row[3],
                    'default': row[4],
                    'balance': row[5],
                    'housing': row[6],
                    'loan': row[7],
                    'contact': row[8],
                    'day': row[9],
                    'month': row[10],
                    'duration': row[11],
                    'campaign': row[12],
                    'pdays': row[13],
                    'previous': row[14],
                    'poutcome': row[15]} 
    X_trainDict.append(example_dict)
    
X_testDict = [] # list of dictionaries 
for row  in X_test:
    example_dict = {'age': row[0],
                    'job': row[1],
                    'marital': row[2],
                    'education': row[3],
                    'default': row[4],
                    'balance': row[5],
                    'housing': row[6],
                    'loan': row[7],
                    'contact': row[8],
                    'day': row[9],
                    'month': row[10],
                    'duration': row[11],
                    'campaign': row[12],
                    'pdays': row[13],
                    'previous': row[14],
                    'poutcome': row[15]} 
    X_testDict.append(example_dict)


################################## Part (a) ###################################
# instantiate AdaBoost class
P2_ada = AdaBoost(X=X_train, data_desc=data_desc, labels=y_train)

# train to 500 iterations
num_iter = 3 # number of AdaBoost iterations (replace with 500)
P2_ada.train(num_iter)

# calculate test/train error at each iteration using AdaBoost final prediction
P2_adaTrainErr = np.zeros([num_iter,])
P2_adaTestErr = np.zeros([num_iter,])
for i in range(num_iter):
    P2_adaTrainErr[i] = P2_ada.adaboost_predict(P2_ada.stumps[:i+1], P2_ada.votes[:i+1], X_train, y_train)[0]
    P2_adaTestErr[i] = P2_ada.adaboost_predict(P2_ada.stumps[:i+1], P2_ada.votes[:i+1], X_test, y_test)[0]

# calculate test/train error of each stump
P2_adaTrainPredictArr =  P2_ada.adaboost_predict(P2_ada.stumps, P2_ada.votes, X_train, y_train)[1]
P2_adaTestPredictArr =  P2_ada.adaboost_predict(P2_ada.stumps, P2_ada.votes, X_test, y_test)[1]

P2_trainArr = np.transpose([y_train] * num_iter)
P2_testArr = np.transpose([y_test] * num_iter)

# -1 where there are prediction errors
train_multArr = np.multiply(P2_adaTrainPredictArr, P2_trainArr)
test_multArr = np.multiply(P2_adaTestPredictArr, P2_testArr)

train_multArrTranspose = train_multArr.transpose()
test_multArrTranspose = test_multArr.transpose()

P2_indivTrainErr = np.zeros([num_iter,])
for i, stump in enumerate(train_multArrTranspose): 
    P2_indivTrainErr[i] = (stump == -1).sum() / num_trainExamples

P2_indivTestErr = np.zeros([num_iter,])
for i, stump in enumerate(test_multArrTranspose): 
    P2_indivTestErr[i] = (stump == -1).sum() / num_testExamples

# print out the results
print("************** PROBLEM 2 PART (a) **************")
print(f"P2_adaTrainErr is {P2_adaTrainErr}")
print(f"P2_adaTestErr is {P2_adaTestErr}")
print(f"P2_indivTrainErr is {P2_indivTrainErr}")
print(f"P2_indivTestErr is {P2_indivTestErr}")
print("")

#%%
################################## Part (b) ###################################
num_bagIters = 3 # number of bagging iterations (replace with 500)

# arrays with rows associated with each data example and columns associated 
# with each iteration of the bagging algorithm. (m,n) values will be the label 
# associated with training example m and the tree at iteration n
bag_trainPredArr = np.zeros([num_trainExamples,num_bagIters])
bag_testPredArr = np.zeros([num_testExamples,num_bagIters])

bag_trainErr = [] # list keeping track of training errors for each iteration
bag_testErr = [] # list keeping track of test errors for each iteration
# run the bagging algorithm
for i in range(num_bagIters):
    # create bootstrap sampled training set
    X_boot = np.zeros(np.shape(X_train)).astype('object')
    y_boot = np.zeros(np.shape(y_train)).astype('int64')
    boot_examples = np.random.choice(np.arange(num_trainExamples), size = num_trainExamples)
    for r in range(num_trainExamples):
        X_boot[r] = X_train[boot_examples[r],:]
        y_boot[r] = y_train[boot_examples[r]]
    
    # train a DT to max depth
    bag_tree =  DT(X=X_boot, data_desc=data_desc, labels=y_boot)
    bag_tree.id3()
    
    # get the train/test predictions for that DT
    for j in range(num_trainExamples):
        bag_trainPredArr[j,i]  = bag_tree.predict(X_trainDict[j])
    for j in range(num_testExamples):
        bag_testPredArr[j,i] = bag_tree.predict(X_testDict[j])
        
    # vote/average the result and track the training/test errors
    curr_bagTrainRes = bag_trainPredArr[:,:i+1]
    curr_bagTestRes = bag_testPredArr[:,:i+1]
    
    curr_bagTrainPred = np.array([1 if (examp_row == 1).sum() >= (i+1)/2 else -1 for examp_row in curr_bagTrainRes])
    curr_bagTestPred = np.array([1 if (examp_row == 1).sum() >= (i+1)/2 else -1 for examp_row in curr_bagTestRes])
    
    # -1 where there are prediction errors
    train_errMult = np.multiply(y_train,curr_bagTrainPred)
    bag_trainErr.append((train_errMult == -1).sum() / num_trainExamples)
    
    test_errMult = np.multiply(y_test,curr_bagTestPred)
    bag_testErr.append((test_errMult == -1).sum() / num_testExamples)

# print out the results
print("************** PROBLEM 2 PART (b) **************")
print(f"P2_bagTrainErr is {bag_trainErr}")
print(f"P2_bagTestErr is {bag_testErr}")
print("")


################################## Part (c) ###################################
num_bags = 3 # number of bagged predictors (replace with 100)
num_bagIters = 3 # number of bagging iterations (replace with 500)

# arrays with rows associated with each data example and columns associated 
# with each iteration of the bagging algorithm. (l,m,n) values will be the 
# label associated with the number of repeats l and training example m and 
# the tree at iteration n
bag_testPredArr = np.zeros([num_bags, num_testExamples, num_bagIters])

# construct bagged predictors
for b in range(num_bags): # for each bagged prediction
    for i in range(num_bagIters): # for each tree in the bagged prediction
        # sample 1000 examples uniformly without replacement
        X_boot = np.zeros([1000,np.size(X_train,1)]).astype('object')
        y_boot = np.zeros(1000).astype('int64')
        boot_examples = np.random.choice(np.arange(num_trainExamples), size = 1000, replace=False)
        for r in range(1000):
            X_boot[r] = X_train[boot_examples[r],:]
            y_boot[r] = y_train[boot_examples[r]]
        
        # train a DT to max depth
        bag_tree =  DT(X=X_boot, data_desc=data_desc, labels=y_boot)
        bag_tree.id3()
        
        # get the test predictions for that DT
        for j in range(num_testExamples):
            bag_testPredArr[b,j,i] = bag_tree.predict(X_testDict[j])

# compute bias, variance, general squared error of single trees
bias_singTreesArr = np.zeros(num_testExamples)
var_singTreesArr = np.zeros(num_testExamples)
for i in range(num_testExamples):
    pred_singTrees = [bag[i,0] for bag in bag_testPredArr] # use first tree
    av_predSingTrees = np.average(pred_singTrees)
    
    bias_singTreesArr[i] = (y_test[i] - av_predSingTrees)**2
    var_singTreesArr[i] = (1/(num_bags-1)) * sum([(pred - av_predSingTrees)**2 for pred in pred_singTrees])

bias_singTrees = np.average(bias_singTreesArr)
var_singTrees = np.average(var_singTreesArr)
gen_sqrErrSingTrees = bias_singTrees + var_singTrees

# compute bias, variance, general squared error of bagged predictors 
bias_baggedArr = np.zeros(num_testExamples)
var_baggedArr = np.zeros(num_testExamples)

# construct the predictions from each bagged predictor
# in a (m,n) array for m training examples and n bagged predictors
bag_predArr = np.zeros([num_testExamples,num_bags])
for i, bag in enumerate(bag_testPredArr): # for each bagged predictor
    bag_predArr[:, i] = np.array([1 if (examp_row == 1).sum() >= num_bagIters/2 else -1 for examp_row in bag])


for i, pred_bag in enumerate(bag_predArr):    
    av_predBagged = np.average(pred_bag)
    
    bias_baggedArr[i] = (y_test[i] - av_predBagged)**2
    var_baggedArr[i] = (1/(num_bags-1)) * sum([(pred - av_predBagged)**2 for pred in pred_bag])

bias_bagged = np.average(bias_baggedArr)
var_bagged = np.average(var_baggedArr)
gen_sqrErrBagged = bias_bagged + var_bagged

# print out the results
print("************** PROBLEM 2 PART (c) **************")
print(f"For Problem 2(c), the single trees have bias = {bias_singTrees:0.4f}, variance = {var_singTrees:0.4f}, general squared error = {gen_sqrErrSingTrees:0.4f}.")
print(f"For Problem 2(c), the bagged predictors have bias = {bias_bagged:0.4f}, variance = {var_bagged:0.4f}, general squared error = {gen_sqrErrBagged:0.4f}.")
print("")


################################## Part (d) ###################################
num_bagIters = 3 # number of bagging iterations (replace with 500)
sizeFeatureSubsetArr = [2, 4, 6] # number of features available to split on

RFbag_trainErrDict = dict()
RFbag_testErrDict = dict()
for sizeFeatureSubset in sizeFeatureSubsetArr:
    RFbag_trainErrDict[sizeFeatureSubset] = list()
    RFbag_testErrDict[sizeFeatureSubset] = list()
    
    # arrays with rows associated with each data example and columns associated 
    # with each iteration of the bagging algorithm. (m,n) values will be the label 
    # associated with training example m and the tree at iteration n
    RFbag_trainPredArr = np.zeros([num_trainExamples,num_bagIters])
    RFbag_testPredArr = np.zeros([num_testExamples,num_bagIters])
    
    RFbag_trainErr = [] # list keeping track of training errors for each iteration
    RFbag_testErr = [] # list keeping track of test errors for each iteration
    # run the bagging algorithm
    for i in range(num_bagIters):
        # create bootstrap sampled training set
        X_boot = np.zeros(np.shape(X_train)).astype('object')
        y_boot = np.zeros(np.shape(y_train)).astype('int64')
        boot_examples = np.random.choice(np.arange(num_trainExamples), size = num_trainExamples)
        for r in range(num_trainExamples):
            X_boot[r] = X_train[boot_examples[r],:]
            y_boot[r] = y_train[boot_examples[r]]
        
        # train a DT to max depth
        RFbag_tree =  DT(X=X_boot, data_desc=data_desc, labels=y_boot)
        RFbag_tree.id3_RF(sizeFeatureSubset)
        
        # get the train/test predictions for that DT
        for j in range(num_trainExamples):
            RFbag_trainPredArr[j,i]  = RFbag_tree.predict(X_trainDict[j])
        for j in range(num_testExamples):
            RFbag_testPredArr[j,i] = RFbag_tree.predict(X_testDict[j])
            
        # vote/average the result and track the training/test errors
        curr_bagTrainRes = RFbag_trainPredArr[:,:i+1]
        curr_bagTestRes = RFbag_testPredArr[:,:i+1]
        
        curr_bagTrainPred = np.array([1 if (examp_row == 1).sum() >= (i+1)/2 else -1 for examp_row in curr_bagTrainRes])
        curr_bagTestPred = np.array([1 if (examp_row == 1).sum() >= (i+1)/2 else -1 for examp_row in curr_bagTestRes])
        
        # -1 where there are prediction errors
        train_errMult = np.multiply(y_train,curr_bagTrainPred)
        RFbag_trainErrDict[sizeFeatureSubset].append((train_errMult == -1).sum() / num_trainExamples)
        
        test_errMult = np.multiply(y_test,curr_bagTestPred)
        RFbag_testErrDict[sizeFeatureSubset].append((test_errMult == -1).sum() / num_testExamples)

# print out the results
print("************** PROBLEM 2 PART (d) **************")
print(f"P2_RFbagTrainErr for feature subset size of 2 is {RFbag_trainErrDict[2]}")
print(f"P2_RFbagTestErr for feature subset size of 2 is {RFbag_testErrDict[2]}")
print(f"P2_RFbagTrainErr for feature subset size of 4 is {RFbag_trainErrDict[4]}")
print(f"P2_RFbagTestErr for feature subset size of 4 is {RFbag_testErrDict[4]}")
print(f"P2_RFbagTrainErr for feature subset size of 6 is {RFbag_trainErrDict[6]}")
print(f"P2_RFbagTestErr for feature subset size of 6 is {RFbag_testErrDict[6]}")
print("")


################################## Part (e) ###################################
num_bags = 3 # number of bagged predictors (replace with 100)
num_bagIters = 3 # number of bagging iterations (replace with 500)
sizeFeatureSubsetArr = [2, 4, 6] # number of features available to split on

RFbag_partE = dict() 
for sizeFeatureSubset in sizeFeatureSubsetArr:
    RFbag_partE[sizeFeatureSubset] = dict() 
    
    # arrays with rows associated with each data example and columns associated 
    # with each iteration of the bagging algorithm. (l,m,n) values will be the 
    # label associated with the number of repeats l and training example m and 
    # the tree at iteration n
    bag_testPredArr = np.zeros([num_bags, num_testExamples, num_bagIters])
    
    # construct bagged predictors
    for b in range(num_bags):
        for i in range(num_bagIters):
            # sample 1000 examples uniformly without replacement
            X_boot = np.zeros([1000,np.size(X_train,1)]).astype('object')
            y_boot = np.zeros(1000).astype('int64')
            boot_examples = np.random.choice(np.arange(num_trainExamples), size = 1000, replace=False)
            for r in range(1000):
                X_boot[r] = X_train[boot_examples[r],:]
                y_boot[r] = y_train[boot_examples[r]]
            
            # train Random Forests to max depth
            bag_tree =  DT(X=X_boot, data_desc=data_desc, labels=y_boot)
            bag_tree.id3_RF(sizeFeatureSubset)
            
            # get the test predictions for that DT
            for j in range(num_testExamples):
                bag_testPredArr[b,j,i] = bag_tree.predict(X_testDict[j])
    
    # compute bias, variance, general squared error of single trees
    bias_singTreesArr = np.zeros(num_testExamples)
    var_singTreesArr = np.zeros(num_testExamples)
    for i in range(num_testExamples):
        pred_singTrees = [bag[i,0] for bag in bag_testPredArr] # use first tree
        av_predSingTrees = np.average(pred_singTrees)
        
        bias_singTreesArr[i] = (y_test[i] - av_predSingTrees)**2
        var_singTreesArr[i] = (1/(num_bags-1)) * sum([(pred - av_predSingTrees)**2 for pred in pred_singTrees])
    
    bias_singTrees = np.average(bias_singTreesArr)
    var_singTrees = np.average(var_singTreesArr)
    gen_sqrErrSingTrees = bias_singTrees + var_singTrees
    
    RFbag_partE[sizeFeatureSubset]['single'] = [bias_singTrees, var_singTrees, gen_sqrErrSingTrees] 
    
    # compute bias, variance, general squared error of bagged predictors 
    bias_baggedArr = np.zeros(num_testExamples)
    var_baggedArr = np.zeros(num_testExamples)
    
    # construct the predictions from each bagged predictor
    # in a (m,n) array for m training examples and n bagged predictors
    bag_predArr = np.zeros([num_testExamples,num_bags])
    for i, bag in enumerate(bag_testPredArr): # for each bagged predictor
        bag_predArr[:, i] = np.array([1 if (examp_row == 1).sum() >= num_bagIters/2 else -1 for examp_row in bag])
    
    
    for i, pred_bag in enumerate(bag_predArr):    
        av_predBagged = np.average(pred_bag)
        
        bias_baggedArr[i] = (y_test[i] - av_predBagged)**2
        var_baggedArr[i] = (1/(num_bags-1)) * sum([(pred - av_predBagged)**2 for pred in pred_bag])
    
    bias_bagged = np.average(bias_baggedArr)
    var_bagged = np.average(var_baggedArr)
    gen_sqrErrBagged = bias_bagged + var_bagged
    
    RFbag_partE[sizeFeatureSubset]['RF'] = [bias_bagged, var_bagged, gen_sqrErrBagged] 

# print out the results
print("************** PROBLEM 2 PART (e) **************")
print(f"For Problem 2(e), for feature subest size 2, the single trees have bias = {RFbag_partE[2]['single'][0]:0.4f}, variance = {RFbag_partE[2]['single'][1]:0.4f}, general squared error = {RFbag_partE[2]['single'][2]:0.4f}.")
print(f"For Problem 2(e), for feature subest size 2, RF bagged predictors have bias = {RFbag_partE[2]['RF'][0]:0.4f}, variance = {RFbag_partE[2]['RF'][1]:0.4f}, general squared error = {RFbag_partE[2]['RF'][2]:0.4f}.")
print(f"For Problem 2(e), for feature subest size 4, the single trees have bias = {RFbag_partE[4]['single'][0]:0.4f}, variance = {RFbag_partE[4]['single'][1]:0.4f}, general squared error = {RFbag_partE[4]['single'][2]:0.4f}.")
print(f"For Problem 2(e), for feature subest size 4, RF bagged predictors have bias = {RFbag_partE[4]['RF'][0]:0.4f}, variance = {RFbag_partE[4]['RF'][1]:0.4f}, general squared error = {RFbag_partE[4]['RF'][2]:0.4f}.")
print(f"For Problem 2(e), for feature subest size 6, the single trees have bias = {RFbag_partE[6]['single'][0]:0.4f}, variance = {RFbag_partE[6]['single'][1]:0.4f}, general squared error = {RFbag_partE[6]['single'][2]:0.4f}.")
print(f"For Problem 2(e), for feature subest size 6, RF bagged predictors have bias = {RFbag_partE[6]['RF'][0]:0.4f}, variance = {RFbag_partE[6]['RF'][1]:0.4f}, general squared error = {RFbag_partE[6]['RF'][2]:0.4f}.")






