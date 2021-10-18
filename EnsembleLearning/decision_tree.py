"""
This will be my implementation of ID3 that I update according to each HW. 

CURRENT UPDATE: HW2
"""

import math
import numpy as np

class DT:
    """Contains all the information about the DT (including features, labels, 
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
        depth: maximum depth of tree. default is infinity. 
         
        
        Values
        __________
        feature_names: list of strings, names of the features 
        feature_categories: list of lists, values the features can take
        label_categories: list of strings/ints, values the labels 
                          can take
        label_categoriesCount: list of ints, how many data instances have each 
                               label
        tree: dictionary, the resulting DTs. keys are integers representing 
              the depth and the values are dictionaries representing the 
              DT for each respective depth key. A DT at each depth tree is 
              itself a dictionary with keys as (feature,value) tuples and 
              values as labels. 
        common_labelsDict: dictionary, stores the most common label for 
                           each subset of data encountered. keys are tuples 
                           of (feature,value) tupes and values are most common
                           labels. 
    """
    
    def __init__(self, X, data_desc, labels, gain_type = 'entropy', depth = math.inf):
        self.X = X
        self.labels = labels
        self.feature_names = list(data_desc.keys())[:-1]
        self.feature_categories = list(data_desc.values())[:-1]
        self.label_categories = list(data_desc.values())[-1]
        self.label_categoriesCount = [list(labels).count(x) for x in self.label_categories]
        self.tree = dict() # initialize as empty dictionary
        self.common_labelDict = dict() # initialize as empty dictionary
        self.gain_type = gain_type
        self.depth = depth
        
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
        
        subset_instancesCount = len(data_ids) # number of instances in data subset
        
        # get the labels for the subset of data specified
        labels = [self.labels[i] for i in data_ids]
        
        # count number of instances for each label in data subset
        labels_count = [labels.count(L) for L in self.label_categories]
        
        # calculate the purity 
        if self.gain_type == 'entropy':
            entropy = sum([-(count/subset_instancesCount) * math.log(count/subset_instancesCount, 2) if count else 0 for count in labels_count])
            return entropy
        elif self.gain_type == 'ME': # majority error
            ME = (subset_instancesCount - max(labels_count)) / subset_instancesCount
            return ME
        elif self.gain_type == 'GI': # Gini Index
            GI = 1 - sum([(count/subset_instancesCount)**2 for count in labels_count])
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
        
        subset_instancesCount = len(data_ids) # number of instances in data subset
        
        # calculate purity of entire data subset
        subset_purity = self._get_purity(data_ids)
        
        # store the values all the instances in the data subset take at this feature
        all_featureValues = [self.X[i][feature_id] for i in data_ids]
        
        # get unique values the data subset take at this feature
        feature_values = self.feature_categories[feature_id]
        
        # count number of instances for each feature value in data subset
        feature_valuesCount = [all_featureValues.count(V) for V in feature_values]
        
        # create a list where the indices correspond to the indices of feature_value; 
        # at each index, have a list that contains the data_ids of instances in the 
        # data subset that take that feature value
        feature_valuesIDs = []
        for feature_value in feature_values: # for each unique feature value
            feature_valueIDs = [] # list of data_ids for that unique feature value
            for i, a in enumerate(all_featureValues): # for array of all feature values
                if a == feature_value:
                    feature_valueIDs.append(data_ids[i])
            feature_valuesIDs.append(feature_valueIDs)
        
        # compute the information gain with the chosen feature
        feature_purity = 0 # initialize feature purity
        for i in range(len(feature_values)): # for each unique feature value
            feature_purity += (feature_valuesCount[i]/subset_instancesCount) \
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
    
    def id3(self):
        """Initializes ID3 algorithm to build DT. Once the DT of maximum depth
        is created, it trims it down to each depth below maximum depth. 

        Returns
        __________
        None
        """
        data_ids = [i for i in range(len(self.X))]
        feature_ids = [i for i in range(len(self.feature_names))]
        branch = []
        self.tree[self.depth] = dict() # initialize as empty dictionary
        self.tree = self._id3_recursive(data_ids, feature_ids, branch, self.tree)
        
        # go through and prune the tree after for all depths below max_depth 
        # (unless maximum depth is infinite)
        if not self.depth == math.inf: 
            for depth in range(self.depth-1,-1,-1): # for all depths below 
                self.tree[depth] = dict() # initialize as empty dictionary
                for key, value in self.tree[depth+1].items(): # look in tree a depth above
                    key_list = list(key)
                    if len(key_list) > depth: # we need to trim this branch
                        key_list.pop() # trim the last node
                        key_trimTup = tuple(key_list) # make it a tuple so it can be dictionary key
                        
                        # find the majority label for that prediciton rule
                        value_trim = self.common_labelDict[key_trimTup]
                        self.tree[depth][key_trimTup] = value_trim
                    else: # doesn't need to be trimmed, keep it as-is
                        self.tree[depth][key] = value                
    
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
        
        # get the most common label among the subset
        subset_labelCategories = list(set(subset_labels)) # set of labels in data subset
        subset_labelCategoriesCount = [subset_labels.count(L) for L in subset_labelCategories]
        subset_mostCommonLabel = subset_labelCategories[subset_labelCategoriesCount.index(max(subset_labelCategoriesCount))]
        
        # for depths below maximum depth, make this node a leaf node and store 
        # the prediction rule in the common label dictionary
        branch_tup = tuple(branch) # make it a tuple so it can be dictionary key
        self.common_labelDict[branch_tup] = subset_mostCommonLabel # add prediction rule to dictionary
        
        # if the maximum depth has been reached, return leaf node with most 
        # common label among the data subset
        # ******************** LEAF NODE ********************
        if curr_depth >= self.depth: 
            tree[self.depth][branch_tup] = subset_mostCommonLabel # add prediction rule to tree
            if len(branch) != 0: # equals 0 when algorithm arrives back at base node
                branch.pop() # pop the last branch out of the list (moving up)
            return tree
            
        # if all instances in subset of data specified by data_ids have the 
        # same label, return leaf node with the label
        # ******************** LEAF NODE ********************
        if len(set(subset_labels)) == 1: 
            tree[self.depth][branch_tup] = self.labels[data_ids[0]] # add prediction rule to tree
            branch.pop() # pop the last branch out of the list (moving up)
            return tree
        
        # if feature_ids is empty (meaning all of the features/attributes
        # have already been used to split), return leaf node with most common
        # label among the data subset
        # ******************** LEAF NODE ********************
        if len(feature_ids) == 0:
            tree[self.depth][branch_tup] = subset_mostCommonLabel # add prediction rule to tree
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
            
            # if child data subset Sv is empty, return leaf node with most 
            # common value of label in data subset S
            # ******************** LEAF NODE ********************
            if len(child_dataIDs) == 0:
                branch_tup = tuple(branch) # make it a tuple so it can be dictionary key
                tree[self.depth][branch_tup] = subset_mostCommonLabel # add prediction rule to tree
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
   
    def id3_RF(self, sizeFeatureSubset):
        """Initializes ID3 algorithm to build DT for Random Forests. The key 
        difference is in the tree construction, where the available attributes
        to split on are limited to the number set by sizeFeatureSubset. 
        Parameters
        __________
        sizeFeatureSubset: int, how many features to limit the Random Forest
                           search to at each node split
        
        Returns
        __________
        None
        """
        data_ids = [i for i in range(len(self.X))]
        feature_ids = [i for i in range(len(self.feature_names))]
        branch = []
        self.tree[self.depth] = dict() # initialize as empty dictionary
        self.tree = self._id3_recursive_RF(data_ids, feature_ids, branch, sizeFeatureSubset, self.tree)    
        
    def _id3_recursive_RF(self, data_ids, feature_ids, branch, sizeFeatureSubset, tree):
        """ID3 algorithm for Random Forests. It is called recursively until 
        some criteria is met. The key difference is in the tree construction, 
        where the available attributes to split on are limited to the number 
        set by sizeFeatureSubset. 
        
        Parameters
        __________
        data_ids: list of ints, identifiers for the elements in the data that 
                  have purity calculated. these are simply just the index of 
                  the elements in the X/y inputs
        feature_ids: list of ints, feature identifiers. these are simply just 
                     the indices of the features in feature_names input
        branch: list of (feature,value) tuples, representation of prediction
                rules that become keys in tree dictionary
        sizeFeatureSubset: int, how many features to limit the Random Forest
                           search to at each node split. 
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
        
        # get the most common label among the subset
        subset_labelCategories = list(set(subset_labels)) # set of labels in data subset
        subset_labelCategoriesCount = [subset_labels.count(L) for L in subset_labelCategories]
        subset_mostCommonLabel = subset_labelCategories[subset_labelCategoriesCount.index(max(subset_labelCategoriesCount))]
        
        # for depths below maximum depth, make this node a leaf node and store 
        # the prediction rule in the common label dictionary
        branch_tup = tuple(branch) # make it a tuple so it can be dictionary key
        self.common_labelDict[branch_tup] = subset_mostCommonLabel # add prediction rule to dictionary
        
        # if the maximum depth has been reached, return leaf node with most 
        # common label among the data subset
        # ******************** LEAF NODE ********************
        if curr_depth >= self.depth: 
            tree[self.depth][branch_tup] = subset_mostCommonLabel # add prediction rule to tree
            if len(branch) != 0: # equals 0 when algorithm arrives back at base node
                branch.pop() # pop the last branch out of the list (moving up)
            return tree
            
        # if all instances in subset of data specified by data_ids have the 
        # same label, return leaf node with the label
        # ******************** LEAF NODE ********************
        if len(set(subset_labels)) == 1: 
            tree[self.depth][branch_tup] = self.labels[data_ids[0]] # add prediction rule to tree
            branch.pop() # pop the last branch out of the list (moving up)
            return tree
        
        # if feature_ids is empty (meaning all of the features/attributes
        # have already been used to split), return leaf node with most common
        # label among the data subset
        # ******************** LEAF NODE ********************
        if len(feature_ids) == 0:
            tree[self.depth][branch_tup] = subset_mostCommonLabel # add prediction rule to tree
            branch.pop() # pop the last branch out of the list (moving up)
            return tree
        
        # otherwise...
        # find feature/attribute that best splits data subset S
        # ********************************************************************
        # ******************* RANDOM FOREST ALTERATION ***********************
        # ********************************************************************
        # we will limit how many features can be split on to sizeFeatureSubset
        # only limit if there are enough featureIDs!
        if len(feature_ids) > sizeFeatureSubset:
            split_featureIDs = list(np.random.choice(feature_ids, size = sizeFeatureSubset, replace = False))
        else: 
            split_featureIDs = feature_ids
        best_featureName, best_featureID = self._get_feature_max_gain(data_ids, split_featureIDs)
        
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
            
            # if child data subset Sv is empty, return leaf node with most 
            # common value of label in data subset S
            # ******************** LEAF NODE ********************
            if len(child_dataIDs) == 0:
                branch_tup = tuple(branch) # make it a tuple so it can be dictionary key
                tree[self.depth][branch_tup] = subset_mostCommonLabel # add prediction rule to tree
                branch.pop() # pop the last branch out of the list (moving up)    
                
            # else...
            # below this branch, add a subtree 
            else: 
                # remove the feature/attribute so it can't be split on again
                if best_featureID in feature_ids:
                    best_featureIDIndex = feature_ids.index(best_featureID)
                    feature_ids.pop(best_featureIDIndex) 

                # recursively call ID3 algorithm
                self.tree = self._id3_recursive_RF(child_dataIDs, feature_ids, branch, sizeFeatureSubset, tree)
            
            # iterate counter. if it equals the number of values in feature_values, 
            # I've exhausted all the values in that feature and need to pop another
            # branch out; also, I need to add that feature ID back in to feature_ids
            counter += 1
            if counter == len(feature_values):
                if len(branch) != 0: # equals 0 when algorithm arrives back at base node
                    branch.pop() # pop the last element out of the list
                    feature_ids.append(best_featureID)  
        return tree

    def predict(self, example_dict, depth=math.inf):
        """Predicts the label of an example. 
        
        Parameters
        __________
        example_dict: dictionary, data example. keys are features/attributes
                      and values are feature/attribute values. 
        depth: int, depth of tree to use in prediction. default is infinity. 
        
        Returns
        __________
        pred_ruleLabel: label prediction for the example
        """
        
        if self.depth == math.inf: # depth wasn't specified when training
            pred_ruleList = list(self.tree[math.inf].keys()) # list of all prediction rules
            for num_splits in range(1000000): # we can't split more than depth we're at
                split_att = pred_ruleList[0][num_splits][0] # attribute being split on
                example_splitAttVal = example_dict[split_att] # finds the value the example takes at that attribute
                
                # get rid of any prediction rule that doesn't have that (feature,value) tuple
                pred_ruleList = [pred for pred in pred_ruleList if pred[num_splits][1] == example_splitAttVal] 
                
                if len(pred_ruleList) == 1: # we found the prediction rule for the example
                    pred_ruleLabel = self.tree[math.inf][pred_ruleList[0]]
                    return pred_ruleLabel
                    
        else: 
            pred_ruleList = list(self.tree[depth].keys()) # list of all prediction rules
            for num_splits in range(depth): # we can't split more than depth we're at
                split_att = pred_ruleList[0][num_splits][0] # attribute being split on
                example_splitAttVal = example_dict[split_att] # finds the value the example takes at that attribute
                
                # get rid of any prediction rule that doesn't have that (feature,value) tuple
                pred_ruleList = [pred for pred in pred_ruleList if pred[num_splits][1] == example_splitAttVal] 
                
                if len(pred_ruleList) == 1: # we found the prediction rule for the example
                    pred_ruleLabel = self.tree[depth][pred_ruleList[0]]
                    return pred_ruleLabel
                
                # print out statement if we don't find the prediction rule for the example
                if num_splits == depth - 1: 
                    raise TypeError("We didn''t find the leaf node for this data example!")
                
        
    














    
    
    
    
    
    
    
    
    
    
