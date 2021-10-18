"""
This will be my implementation of AdaBoost for HW2. 
"""

import math
import numpy as np
import time

class AdaBoost:
    """Contains all the information about the AdaBoost (including features, labels, 
    gain type, tree structure).
    
    Attributes
    __________
        
        Parameters
        __________
        X: array, all the data inputs (e.g., for n data points with
           m features/attributes, X will be nxm array)
        data_desc: dictionary, the data description. keys are feature/label 
                   names (with label being last in order) and values are 
                   lists of possible values they can take.
        labels: array, all the  data labels (e.g., for n data points, 
                y will be nx1 array)
        gain_type: gain type being used in gain calculations. default 
                   is entropy. 
                   * gain types currently implemented: entropy, ME, GI
        depth: maximum depth of tree (weak classifier). default is 1. 
         
        
        Values
        __________
        feature_names: list of strings, names of the features 
        feature_categories: list of lists, values the features can take
        label_categories: list of strings/ints, values the labels 
                          can take
        label_categoriesCount: list of ints, how many data instances have each 
                               label
        data_ids: list of ints, identifier of each data example in X
        feature_ids: list of ints, identifier of each feature in feature_names
        feature_valuesIDs: list of list of lists, the data ids that correspond 
                                 to the labels in label_categories
        label_categoriesDataIDs: list of lists, the data ids that correspond 
                                 to the labels in label_categories
        sample_weights: numpy array, sample weights
        tree: dictionary, the current DT. keys are (feature,value) tuples and 
              values are labels. 
        stumps: list of DTs, the DT for each iteration of AdaBoost
        votes: list of floats, the votes for each iteration of AdaBoost
    """
    
    def __init__(self, X, data_desc, labels, gain_type = 'entropy', depth = 1):
        self.X = X
        self.labels = labels
        self.feature_names = list(data_desc.keys())[:-1]
        self.feature_categories = list(data_desc.values())[:-1]
        self.label_categories = list(data_desc.values())[-1]        
        self.label_categoriesCount = [list(labels).count(x) for x in self.label_categories]
        self.data_ids = [i for i in range(len(self.X))]
        self.feature_ids = [i for i in range(len(self.feature_names))]
        
        self.feature_valuesIDs = list()
        for feature_id in self.feature_ids:
            # store the values all the instances in the data subset take at this feature
            all_featureValues = list(self.X[:,feature_id])
            feature_valuesIDs = list()
            for feature_value in self.feature_categories[feature_id]: # for each unique feature value
                feature_valueIDs = list() # list of data_ids for that unique feature value
                for i, a in enumerate(all_featureValues): # for array of all feature values
                    if a == feature_value:
                        feature_valueIDs.append(self.data_ids[i])
                feature_valuesIDs.append(feature_valueIDs)
            self.feature_valuesIDs.append(feature_valuesIDs)
        
        self.label_categoriesDataIDs = list()
        for c in self.label_categories: 
            label_categoryDataIDs = [i for i,l in enumerate(self.labels) if l==c]
            self.label_categoriesDataIDs.append(label_categoryDataIDs)
        
        self.sample_weights = np.array([1/(len(self.X)) for i in range(len(self.X))]) # initialize 
        self.tree = dict() # initialize as empty dictionary
        self.gain_type = gain_type
        self.depth = depth
        
        self.stumps = list() # initialize as empty list of stumps
        self.votes = list() # initialize as empty numpy array of votes
        
    def _get_purity(self, data_ids):
        """Calculates purity for the subset of the data specified by 
        data_ids. 
        
        Parameters
        __________
        data_ids: list of ints, identifiers for the elements in the data that 
                  have purity calculated. these are simply just the index of 
                  the elements in the X/y inputs
                  
        Return 
        __________
        purity: float, purity calculation (entropy, ME, GI)
        """

        # if the data subset is empty, return 0
        if len(data_ids) == 0:
            return 0
        
        # sum up the fractional count for the subset
        subset_fracCount = sum([self.sample_weights[i] for i in data_ids]) 
        
        # sum up the fractional count for all labels in label_categories
        label_categoriesFracCount = list()
        for c in self.label_categoriesDataIDs:
            label_categoriesFracCount.append(sum([self.sample_weights[i] for i in data_ids if i in c]))
        
        # calculate the purity 
        if self.gain_type == 'entropy':
            entropy = sum([-(count/subset_fracCount) * math.log(count/subset_fracCount, 2) if count else 0 for count in label_categoriesFracCount])
            return entropy
        elif self.gain_type == 'ME': # majority error
            ME = (subset_fracCount - max(label_categoriesFracCount)) / subset_fracCount
            return ME
        elif self.gain_type == 'GI': # Gini Index
            GI = 1 - sum([(count/subset_fracCount)**2 for count in label_categoriesFracCount])
            return GI
        else:
            raise ValueError("This type of purity measure is not currently implemented.")
   
    def _get_gain(self, data_ids, feature_id):
        """Calculates the gain for a given feature and subset of data 
        specified by data_ids. 
        
        Parameters
        __________
        data_ids: list of ints, identifiers for the elements in the data that 
                  have purity calculated. these are simply just the index of 
                  the elements in the X/y inputs
        feature_id: int, feature identifier. this is simply just the index 
                    of the feature in feature_names input
        
        Returns
        __________
        gain: float, the gain for a given feature
        """

        subset_fracCount = 1
        
        # calculate purity of entire data subset
        subset_purity = self._get_purity(data_ids)
        
        # get unique values the data subset take at this feature
        feature_values = self.feature_categories[feature_id]      
        feature_valuesIDs = self.feature_valuesIDs[feature_id]
        
        # sum up the fractional count for each feature value in data subset
        feature_valuesFracCount = list()
        for f in feature_valuesIDs:
            feature_valuesFracCount.append(sum([self.sample_weights[i] for i in data_ids if i in f]))
        
        # compute the information gain with the chosen feature
        feature_purity = 0 # initialize feature purity
        for i in range(len(feature_values)): # for each unique feature value
            feature_purity += (feature_valuesFracCount[i]/subset_fracCount) \
                * self._get_purity(feature_valuesIDs[i])
        
        gain = subset_purity - feature_purity
        return gain
    
    def _get_feature_max_gain(self, data_ids, feature_ids):
        """Finds the feature/attribute in feature_ids that maximizes the 
        gain for subset of data specified by data_ids. 
        
        Parameters
        __________
        data_ids: list of ints, identifiers for the elements in the data that 
                  have purity calculated. these are simply just the index of 
                  the elements in the X/y inputs
        feature_ids: list of ints, feature identifiers. these are simply just 
                     the indices of the features in feature_names input
        
        Returns
        __________
        self.feature_names[max_featureID]: string, name of feature that 
                                           maximizes the gain
        max_featureID: int, feature identifier of the feature that maximizes
                       the gain
        """

        # sort the feature_ids
        feature_ids.sort()
        
        # get the gain for each feature in feature_ids
        feature_gains = [self._get_gain(data_ids, feature_id) for feature_id in feature_ids]
        feature_gains = [round(feature_gain,8) for feature_gain in feature_gains]
        max_featureGain = max(feature_gains)
        
        # find the feature that maximizes the gain
        # in case of tie, gives first index with the maximum gain
        max_featureID = feature_ids[feature_gains.index(max_featureGain)] 

        return self.feature_names[max_featureID], max_featureID
    
    def _id3(self):
        """Initializes ID3 algorithm to build DT. Once the DT of maximum depth
        is created, it trims it down to each depth below maximum depth. 

        Returns
        __________
        None
        """
        branch = []
        self.tree = self._id3_recursive(self.data_ids, self.feature_ids, branch, self.tree)
    
    def _id3_recursive(self, data_ids, feature_ids, branch, tree):
        """ID3 algorithm. It is called recursively until some criteria is met.
        
        Parameters
        __________
        data_ids: list of ints, identifiers for the elements in the data that 
                  have purity calculated. these are simply just the index of 
                  the elements in the X/y inputs
        feature_ids: list of ints, feature identifiers. these are simply just 
                     the indices of the features in feature_names input
        branch: list of (feature,value) tuples, representation of prediction
                rules that become keys in tree dictionary
        tree: (recursive algorithm, see return)
        
        Returns
        __________
        tree: dictionary, the resulting DT. keys are tuples of 
              (feature,value) tuples and values are labels.
        """
            
        # find the current depth
        curr_depth = len(branch)
        
        # generate (ordered) list of labels corresponding to subset of data 
        # specified by data_ids
        subset_labels = [self.labels[i] for i in data_ids]
        
        # get the label with the largest fractional count among the subset
        subset_labelCategories = self.label_categories # set of labels in data subset
        label_categoriesFracCount = list()
        for c in self.label_categoriesDataIDs:
            label_categoriesFracCount.append(sum([self.sample_weights[i] for i in data_ids if i in c]))
        subset_largestFracCountLabel = subset_labelCategories[label_categoriesFracCount.index(max(label_categoriesFracCount))]

        branch_tup = tuple(branch) # make it a tuple so it can be dictionary key
        
        # if the maximum depth has been reached, return leaf node with  
        # the largest fractional count among the data subset
        # ******************** LEAF NODE ********************
        if curr_depth >= self.depth: 
            tree[branch_tup] = subset_largestFracCountLabel # add prediction rule to tree
            if len(branch) != 0: # equals 0 when algorithm arrives back at base node
                branch.pop() # pop the last branch out of the list (moving up)
            return tree
            
        # if all instances in subset of data specified by data_ids have the 
        # same label, return leaf node with the label
        # ******************** LEAF NODE ********************
        if len(set(subset_labels)) == 1: 
            tree[branch_tup] = self.labels[data_ids[0]] # add prediction rule to tree
            branch.pop() # pop the last branch out of the list (moving up)
            return tree
        
        # if feature_ids is empty (meaning all of the features/attributes
        # have already been used to split), return leaf node with 
        # the largest fractional count among the data subset
        # ******************** LEAF NODE ********************
        if len(feature_ids) == 0:
            tree[branch_tup] = subset_largestFracCountLabel # add prediction rule to tree
            branch.pop() # pop the last branch out of the list (moving up)
            return tree
        
        # otherwise...
        # find feature/attribute that best splits data subset S
        best_featureName, best_featureID = self._get_feature_max_gain(data_ids, feature_ids)
        
        # find all the values the splitting feature/attribute can take and 
        # create branches corresponding to each
        feature_values = self.feature_categories[best_featureID]
                
        # loop through all the branches and create child nodes
        counter = 0 # counter local to the feature being looped through
        for feature_value in feature_values:
            branch.append((best_featureName,feature_value)) # append tuple of (feature,value) pair

            # find child subset of data Sv in data subset S with 
            # feature/attribute = feature_value
            child_dataIDs = [n for n in data_ids if self.X[n][best_featureID] == feature_value]
            
            # if child data subset Sv is empty, return leaf node with  
            # the largest fractional count in data subset S
            # ******************** LEAF NODE ********************
            if len(child_dataIDs) == 0:
                branch_tup = tuple(branch) # make it a tuple so it can be dictionary key
                tree[branch_tup] = subset_largestFracCountLabel # add prediction rule to tree
                branch.pop() # pop the last branch out of the list (moving up)    
                
            # else...
            # below this branch, add a subtree 
            else: 
                # remove the feature/attribute so it can't be split on again
                if best_featureID in feature_ids:
                    best_featureIDIndex = feature_ids.index(best_featureID)
                    feature_ids.pop(best_featureIDIndex) 

                # recursively call ID3 algorithm
                self.tree = self._id3_recursive(child_dataIDs, feature_ids, branch, tree)
            
            # iterate counter. if it equals the number of values in feature_values, 
            # I've exhausted all the values in that feature and need to pop another
            # branch out; also, I need to add that feature ID back in to feature_ids
            counter += 1
            if counter == len(feature_values):
                if len(branch) != 0: # equals 0 when algorithm arrives back at base node
                    branch.pop() # pop the last element out of the list
                    feature_ids.append(best_featureID)  
        return tree
    
    def train(self, num_iters):
        """Trains AdaBoost to the specified number of iterations, after which 
        a collection of weak classifiers (stumps) and votes will be filled. 
        
        Parameters
        __________
        num_iters: int, number of iterations to run AdaBoost
        
        Returns
        __________
        None
        """
        for t in range(num_iters):
            # find the classifier h_t
            self.feature_ids = [i for i in range(len(self.feature_names))]
            self.tree = dict() # initialize as empty dictionary
            self._id3()
            self.stumps.append(self.tree) # add the classifier to the stumps list
            
            # compute that classifier's vote
            pred_err, pred_labels = self._train_predictError()
            vote = 0.5 * math.log((1-pred_err)/pred_err)
            self.votes.append(vote)
            
            # update the values of the weights 
            unnorm_sampleWeights = self.sample_weights * np.exp(-vote * self.labels * pred_labels) 
            self.sample_weights = unnorm_sampleWeights / sum(unnorm_sampleWeights)
        self.votes = np.array(self.votes)

    def _train_predictError(self):
        """Trains AdaBoost to the specified number of iterations, after which 
        a collection of weak classifiers (stumps) and votes will be filled. 
        
        Parameters
        __________
        None
        
        Returns
        __________
        pred_err: float, weighted training error
        pred_labels: numpy array, prediction labels 
        """
        pred_labels = []
        pred_err = 0
        pred_ruleList = list(self.tree.keys()) # list of all prediction rules
        split_att = pred_ruleList[0][0][0] # attribute being split on
        split_attFeatureID = self.feature_names.index(split_att) # feature id of attribute being split on
        
        for i, example in enumerate(self.X): # for each example
            example_splitAttVal = example[split_attFeatureID] # finds the value the example takes at that attribute
            
            # get rid of any prediction rule that doesn't have that (feature,value) tuple
            pred_ruleListExample = [pred for pred in pred_ruleList if pred[0][1] == example_splitAttVal] 
            
            pred_ruleLabel = self.tree[pred_ruleListExample[0]]
            pred_labels.append(pred_ruleLabel)
            
            # if there is a prediction error, add the weight to the prediction error
            if pred_ruleLabel != self.labels[i]: # prediction error
                pred_err += self.sample_weights[i]
        
        pred_labels = np.array(pred_labels)
        return pred_err, pred_labels

    def adaboost_predict(self, stumps, votes, X, y):
        """Returns the final hypothesis prediction of Adaboost for a given 
        array of examples. 
        
        Parameters
        __________
        stumps: list of DTs, the DT for each iteration of AdaBoost
        votes: array, the votes for each iteration of AdaBoost
        X: array, all the data inputs (e.g., for n data points with
           m features/attributes, X will be nxm array)
        y: array, all the  data labels (e.g., for n data points, 
                y will be nx1 array)
        
        Returns
        __________
        ada_err: float, training error in comparison to y
        """
        
        num_examples = len(X)
        num_stumps = len(stumps)
        num_err = 0
        
        # create a (m,t) array of predictions 
        predict_array = np.zeros((num_examples,num_stumps))
                                 
        for j, stump in enumerate(stumps): # for each column
            pred_ruleList = list(stump.keys()) # list of all prediction rules
            split_att = pred_ruleList[0][0][0] # attribute being split on
            split_attFeatureID = self.feature_names.index(split_att)
            
            for i, example in enumerate(X): # for each row
                example_splitAttVal = example[split_attFeatureID] # finds the value the example takes at that attribute
                
                # get rid of any prediction rule that doesn't have that (feature,value) tuple
                pred_ruleListExample = [pred for pred in pred_ruleList if pred[0][1] == example_splitAttVal] 
                
                pred_ruleLabel = stump[pred_ruleListExample[0]]
                predict_array[i,j] = pred_ruleLabel
        
        ada_labels = []
        for i, predict_row in enumerate(predict_array):
            ada_sum = sum(votes * predict_row)
            if ada_sum >= 0: # positive
                ada_labels.append(1)
                if y[i] != 1: # prediction error
                    num_err += 1
                
            else: # negative
                ada_labels.append(-1)
                if y[i] != -1: # prediction error
                    num_err += 1
                    
        ada_err = num_err / num_examples
        
        return ada_err, predict_array
    
    
    
    
    
    
    
    
    
    
