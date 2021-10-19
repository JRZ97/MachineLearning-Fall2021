# Decision Tree
This is an ID3 Decision Tree learning algorithm used in HW1. We used three types of purity in this implementation: entropy, Majority Error (ME), and Gini Index (GI). 
## Dataset Folders
### Car
This dataset is from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/car+evaluation). Data descriptions are available in "data-desc.txt". 
### Bank
This dataset is from [UCI repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). Data descriptions are available in "data-desc.txt". 
## Code
### "HW1.py", "run.sh"
This code is specific to running the tasks specified in HW1 (i.e., reporting in tables the avrage prediction errors on the Car/Bank datasets using different measures of purity. To run my code and get printed outputs, use the following lines in the terminal while in the directory "DecisionTree": 
```
chmod u+x run.sh 
./run.sh
```
### "decision_tree.py"
This is the implementation of the ID3 Decision Tree learning algorithm. Using it defines an instance of the class "DT" that contains methods that run the ID3 algorithm and predict the label of an example. Here is a description of an instance of the class "DT": 
   

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

Running ID3 on an instance of this class is done using the "id3()" method, i.e., "DT.id3()". When ran, this creates decision trees with a specified purity measure and maximum depth that can be accessed by, for example, "DT.tree[depth]".  
