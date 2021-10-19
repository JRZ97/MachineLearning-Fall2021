# Ensemble Learning
This is the implementation of the ensemble learning algorithms used in HW2. These consist of: part (a) Adaboost, parts (b) and (c) bagged trees, parts (d) and (e) random forests. 
 
## Dataset Folders
### Bank-1
This dataset is from [UCI repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). Data descriptions are available in "data-desc.txt". 
## Slurm Outputs Folders
### Slurm outputs
These are my slurm outputs for each part of Problem 2. These were run on CHPC and had runtimes listed in my LaTeX homework. 
## Code
### "ensemble.py", "run.sh"
This code is specific to running the tasks specified in HW2 (i.e., AdaBoost, bagged trees, and random forests training/test errors; bagged trees and random forests bias and variance). To run my code and get printed outputs, use the following lines in the terminal in the directory "EnsembleLearning": 
```
chmod u+x run.sh 
./run.sh
```
** IMPORTANT **
Due to having very inefficient runtimes, I have greatly reduced these problems to (a) 3 AdaBoost iterations (b) 3 trees (c) 3 bagged predictors with 3 trees each (d) 3 trees for each feature subset size (e) 3 bagged predictors with 3 trees each. These will output print statements so that you can check that my code is running without errors. If you desire to adjust these values, here are the lines in the code to adjust how many iterations and number of trees if you desire: 
(a) line 139, num_iters (adjusts number of AdaBoost iterations)
(b) line 181, num_bagIters (adjusts number of trees for bagged trees algorithm)
(c) lines 233 and 234, num_bags and num_bagIters (adjusts number of bagged predictors and number of trees in each predictor for bagged trees algorithm, respectively)
(d) line 304, num_bagIters (adjusts number of trees for random forests algorithm)
(e) lines 367 and 368, num_bags and num_bagIters (adjusts number of bagged predictors and number of trees in each predictor for random forests algorithm, respectively)
### "decision_tree.py"
This is the implementation of the ID3 Decision Tree learning algorithm with additional methods for the random forests algorithm. Using it defines an instance of the class "DT" that contains methods that run the ID3 algorithm and predict the label of an example. Here is a description of an instance of the class "DT": 
   

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
        label_categoriesCount: list of ints, how many data instances have 
                               each label. 
        tree: dictionary, the resulting DTs. keys are integers representing 
              the depth and the values are dictionaries representing the 
              DT for each respective depth key. A DT at each depth tree is 
              itself a dictionary with keys as (feature,value) tuples and 
              values as labels. 
        common_labelsDict: dictionary, stores the most common label for 
                           each subset of data encountered. keys are tuples 
                           of (feature,value) tupes and values are most 
                           common labels. 

Running ID3 for random forests on an instance of this class is done using the "id3_RF(sizeFeatureSubset)" method, i.e., "DT.id3(sizeFeatureSubset)", where sizeFeatureSubset is the number of features available to split on at each node. When ran, this creates decision trees with a specified purity measure and depth that can be accessed by, for example, "DT.tree[depth]". When growing to fully expanded decision trees, depth is not specified and defaults to infinity. 
### "AdaBoost.py"
This is the implementation of the AdaBoost learning algorithm. Using it defines an instance of the class "AdaBoost" that contains methods that run the learning algorithm and predict the label of an example. Here is a description of an instance of the class "AdaBoost": 
  
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

Clearly, this is just a modification of the DT class that can be integrated in the future for a global class; the key difference is in the inclusion of handling fractional examples in the gain calculation and eventual tree construction. 
Running the Adaboost algorithm on an instance of this class is done using the "train(num_iter)" method, i.e., "AdaBoost.train(num_iter)", where num_iter is the number of iterations to run the algorithm to. When ran, this creates decision stumps and a votes array that can be accessed by, for example, AdaBoost.stumps and AdaBoost.votes. 
