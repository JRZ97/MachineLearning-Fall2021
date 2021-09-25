"""
This is the script that will run through HW1 problems. 
"""

import numpy as np
import pandas as pd
import csv
import math
from decision_tree import DT # import decision tree class

gains = ['entropy', 'ME', 'GI'] # gain types

# =============================================================================
# Problem 2
# =============================================================================
P2_maxDepth = 6
P2_index = ['dep_' + str(i) for i in range(1,P2_maxDepth+1)]
P2_index.append('avg') # results row labels

# read in data dictionary with possible values
data_desc = {
    'buying': ['vhigh', 'high', 'med', 'low'],
    'maint': ['vhigh', 'high', 'med', 'low'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low','med','high'],
    'eval': ['unacc', 'acc', 'good', 'vgood']}

trainData_df = pd.read_csv('car/train.csv', sep=',', names=data_desc.keys())
testData_df = pd.read_csv('car/test.csv', sep=',', names=data_desc.keys())

# separate target from predictors
X = np.array(trainData_df.drop('eval', axis=1).copy())
y = np.array(trainData_df['eval'].copy())

P2_trainErrCount = np.zeros((P2_maxDepth,3))
P2_testErrCount = np.zeros((P2_maxDepth,3))

# create dictionary with all the tree objects for Problem 2
P2 = dict()

for g in gains:
    P2[g] = DT(X=X, data_desc=data_desc, labels=y, gain_type = g, depth = P2_maxDepth)
    P2[g].id3() # run algorithm id3 to build a tree
    
for d in range(1,P2_maxDepth):
    for g in gains: 
        if P2[g].tree[d] == P2[g].tree[d+1]:
            print('P2 depth of ' + d + ' for gain ' + g + ' not needed.')
    
# get training prediction error
for index,row in trainData_df.iterrows(): # for each training data example
    data_set = {('buying',row['buying']),
                ('maint',row['maint']),
                ('doors',row['doors']),
                ('persons',row['persons']),
                ('lug_boot',row['lug_boot']),
                ('safety',row['safety'])} # create set of (feature,value) tuples
    data_label = row['eval']
    for i,g in enumerate(gains):
        for depth in range(1,P2_maxDepth+1):
            for key in P2[g].tree[depth].keys(): # for each key in the tree dictionary  
                key_set = set(key) # gets set of (feature,value) pairs
                key_label = P2[g].tree[depth][key]
                if key_set.issubset(data_set): # this identifies leaf node for that data example
                    if data_label != key_label: # prediction error
                        P2_trainErrCount[depth-1,i-1] += 1
                    break # leaf node already identified 

P2_trainErr = np.round(P2_trainErrCount/len(trainData_df), decimals = 4)
P2_trainHeurAv = np.round(np.average(a=P2_trainErr, axis=0), decimals = 4) # average over heuristics
P2_trainErr = np.vstack((P2_trainErr,P2_trainHeurAv))

P2_trainErrDF = pd.DataFrame(P2_trainErr, columns=gains, index=P2_index)
print('P2 training error:')
print(P2_trainErrDF)

# get test prediction error
for index,row in testData_df.iterrows(): # for each training data example
    data_set = {('buying',row['buying']),
                ('maint',row['maint']),
                ('doors',row['doors']),
                ('persons',row['persons']),
                ('lug_boot',row['lug_boot']),
                ('safety',row['safety'])} # create set of (feature,value) tuples
    data_label = row['eval']
    for i,g in enumerate(gains):
        for depth in range(1,P2_maxDepth+1):
            for key in P2[g].tree[depth].keys(): # for each key in the tree dictionary  
                key_set = set(key) # gets set of (feature,value) pairs
                key_label = P2[g].tree[depth][key]
                if key_set.issubset(data_set): # this identifies leaf node for that data example
                    if data_label != key_label: # prediction error
                        P2_testErrCount[depth-1,i-1] += 1
                    break # leaf node already identified 
P2_testErr = np.round(P2_testErrCount/len(testData_df), decimals = 4)
P2_testHeurAv = np.round(np.average(a=P2_testErr, axis=0), decimals = 4) # average over heuristics
P2_testErr = np.vstack((P2_testErr,P2_testHeurAv))

P2_testErrDF = pd.DataFrame(P2_testErr, columns=gains, index=P2_index)
print('')
print('P2 test error:')
print(P2_testErrDF)


# =============================================================================
# Problem 3
# =============================================================================
P3_maxDepth = 16
P3_index = ['dep_' + str(i) for i in range(1,P3_maxDepth+1)]
P3_index.append('avg') # results row labels


################################## Part (a) ###################################

# go through and convert numeric features to binary, leaving unknown as an 
# attribute value. 
with open('bank/train.csv', newline='') as f:
    reader = csv.reader(f)
    P3_train_prelimData = np.array(list(reader))
    
with open('bank/test.csv', newline='') as f:
    reader = csv.reader(f)
    P3_test_prelimData = np.array(list(reader))

P3_trainData = P3_train_prelimData.copy() # modified categorical array
P3_testData = P3_test_prelimData.copy() # modified categorical array

column_defaultTrain = P3_train_prelimData[:,4]
column_defaultTest = P3_test_prelimData[:,4]

categ_columns = [0,5,9,11,12,13,14]
for column in categ_columns:
    columnTrain_prelim = P3_train_prelimData[:,column].astype(float)
    columnTest_prelim = P3_test_prelimData[:,column].astype(float)
    med_valTrain = np.median(columnTrain_prelim)

    columnTrain_categ = []
    for x in columnTrain_prelim:
        if x >= med_valTrain:
            columnTrain_categ.append('High')
        else:
            columnTrain_categ.append('Low')
    P3_trainData[:,column] = columnTrain_categ

    columnTest_categ = []
    for x in columnTest_prelim:
        if x >= med_valTrain:
            columnTest_categ.append('High')
        else:
            columnTest_categ.append('Low')
    P3_testData[:,column] = columnTest_categ

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
    'y': ['yes','no']}

trainData_df = pd.DataFrame(P3_trainData,columns=data_desc.keys())
testData_df = pd.DataFrame(P3_testData,columns=data_desc.keys())

# separate target from predictors
X = np.array(trainData_df.drop('y', axis=1).copy())
y = np.array(trainData_df['y'].copy())

P3_a_trainErrCount = np.zeros((P3_maxDepth,3))
P3_a_testErrCount = np.zeros((P3_maxDepth,3))

# create dictionary with all the tree objects for Problem 3
P3_a = dict()

for g in gains:
    P3_a[g] = DT(X=X, data_desc=data_desc, labels=y, gain_type = g, depth = P3_maxDepth)
    P3_a[g].id3() # run algorithm id3 to build a tree
    
for d in range(1,P3_maxDepth):
    for g in gains: 
        if P3_a[g].tree[d] == P3_a[g].tree[d+1]:
            print('P3(a) depth of ' + d + ' for gain ' + g + ' not needed.')

# get training prediction error
for row in P3_trainData: # for each training data example
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
                    'poutcome': row[15]} # create dictionary for feature 
    example_label = row[16]
    
    for i,g in enumerate(gains):
        for depth in range(1,P3_maxDepth+1):
            pred_label = P3_a[g].predict(example_dict,depth)
            if example_label != pred_label: # prediction error
                P3_a_trainErrCount[depth-1,i-1] += 1
                    
P3_a_trainErr = np.round(P3_a_trainErrCount/len(trainData_df), decimals = 4)
P3_a_trainHeurAv = np.round(np.average(a=P3_a_trainErr, axis=0), decimals = 4) # average over heuristics
P3_a_trainErr = np.vstack((P3_a_trainErr,P3_a_trainHeurAv))

P3_a_trainErrDF = pd.DataFrame(P3_a_trainErr, columns=gains, index=P3_index)

print('')
print('P3(a) training error:')
print(P3_a_trainErrDF)

# get test prediction error
for row in P3_testData: # for each training data example
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
                    'poutcome': row[15]} # create dictionary for feature 
    example_label = row[16]
    
    for i,g in enumerate(gains):
        for depth in range(1,P3_maxDepth+1):
            pred_label = P3_a[g].predict(example_dict,depth)
            if example_label != pred_label: # prediction error
                P3_a_testErrCount[depth-1,i-1] += 1
            
P3_a_testErr = np.round(P3_a_testErrCount/len(testData_df), decimals = 4)
P3_a_testHeurAv = np.round(np.average(a=P3_a_testErr, axis=0), decimals = 4) # average over heuristics
P3_a_testErr = np.vstack((P3_a_testErr,P3_a_testHeurAv))

P3_a_testErrDF = pd.DataFrame(P3_a_testErr, columns=gains, index=P3_index)
print('')
print('P3(a) test error:')
print(P3_a_testErrDF)


################################## Part (b) ###################################

# read in data dictionary with possible values
data_desc = {
    'age': ['High','Low'],
    'job': ['admin.','unemployed','management','housemaid',
            'entrepreneur','student','blue-collar','self-employed',
            'retired','technician','services'],
    'marital': ['married','divorced','single'],
    'education': ['secondary','primary','tertiary'],
    'default': ['yes','no'],
    'balance': ['High','Low'],
    'housing': ['yes','no'],
    'loan': ['yes','no'],
    'contact': ['telephone','cellular'],
    'day': ['High','Low'],
    'month': ['jan','feb','mar','apr','may','jun','jul','aug',
              'sep','oct','nov','dec'],
    'duration': ['High','Low'],
    'campaign': ['High','Low'],
    'pdays': ['High','Low'],
    'previous': ['High','Low'],
    'poutcome': ['other','failure','success'],
    'y': ['yes','no']}

P3_trainDataUnk = P3_trainData.copy() # modified unknown array
P3_testDataUnk = P3_testData.copy() # modified unknown array

column_atts = ['job','education','contact','poutcome']
unk_columns = [1,3,8,15]
for index, column in enumerate(unk_columns):
    columnTrain_prelim = list(P3_trainData[:,column])
    columnTest_prelim = list(P3_testData[:,column])
    
    column_values = data_desc[column_atts[index]]
    
    column_valuesTrainCount = [columnTrain_prelim.count(v) for v in column_values]
    column_TrainmostCommonValue = column_values[column_valuesTrainCount.index(max(column_valuesTrainCount))]
    
    column_valuesTestCount = [columnTest_prelim.count(v) for v in column_values]
    column_TestmostCommonValue = column_values[column_valuesTestCount.index(max(column_valuesTestCount))]
    
    for i,x in enumerate(columnTrain_prelim):
        if x == 'unknown':
            columnTrain_prelim[i] = column_TrainmostCommonValue
    P3_trainDataUnk[:,column] = columnTrain_prelim

    for i,x in enumerate(columnTest_prelim):
        if x == 'unknown':
            columnTest_prelim[i] = column_TestmostCommonValue
    P3_testDataUnk[:,column] = columnTest_prelim

trainData_df = pd.DataFrame(P3_trainDataUnk,columns=data_desc.keys())
testData_df = pd.DataFrame(P3_testDataUnk,columns=data_desc.keys())

# separate target from predictors
X = np.array(trainData_df.drop('y', axis=1).copy())
y = np.array(trainData_df['y'].copy())

P3_b_trainErrCount = np.zeros((P3_maxDepth,3))
P3_b_testErrCount = np.zeros((P3_maxDepth,3))

# create dictionary with all the tree objects for Problem 3
P3_b = dict()

for g in gains:
    P3_b[g] = DT(X=X, data_desc=data_desc, labels=y, gain_type = g, depth = P3_maxDepth)
    P3_b[g].id3() # run algorithm id3 to build a tree
    
for d in range(1,P3_maxDepth):
    for g in gains: 
        if P3_b[g].tree[d] == P3_b[g].tree[d+1]:
            print('P3(b) depth of ' + d + ' for gain ' + g + ' not needed.')

# get training prediction error
for row in P3_trainDataUnk: # for each training data example
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
                    'poutcome': row[15]} # create dictionary for feature 
    example_label = row[16]
    
    for i,g in enumerate(gains):
        for depth in range(1,P3_maxDepth+1):
            pred_label = P3_b[g].predict(example_dict,depth)
            if example_label != pred_label: # prediction error
                P3_b_trainErrCount[depth-1,i-1] += 1
              
P3_b_trainErr = np.round(P3_b_trainErrCount/len(trainData_df), decimals = 4)
P3_b_trainHeurAv = np.round(np.average(a=P3_b_trainErr, axis=0), decimals = 4) # average over heuristics
P3_b_trainErr = np.vstack((P3_b_trainErr,P3_b_trainHeurAv))

P3_b_trainErrDF = pd.DataFrame(P3_b_trainErr, columns=gains, index=P3_index)
print('')
print('P3(b) training error:')
print(P3_b_trainErrDF)

# get test prediction error
for row in P3_testDataUnk: # for each training data example
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
                    'poutcome': row[15]} # create dictionary for feature 
    example_label = row[16]
    
    for i,g in enumerate(gains):
        for depth in range(1,P3_maxDepth+1):
            pred_label = P3_b[g].predict(example_dict,depth)
            if example_label != pred_label: # prediction error
                P3_b_testErrCount[depth-1,i-1] += 1
                
P3_b_testErr = np.round(P3_b_testErrCount/len(testData_df), decimals = 4)
P3_b_testHeurAv = np.round(np.average(a=P3_b_testErr, axis=0), decimals = 4) # average over heuristics
P3_b_testErr = np.vstack((P3_b_testErr,P3_b_testHeurAv))

P3_b_testErrDF = pd.DataFrame(P3_b_testErr, columns=gains, index=P3_index)
print('')
print('P3(b) test error:')
print(P3_b_testErrDF)














     