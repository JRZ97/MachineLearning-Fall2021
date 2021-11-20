# SVM
This is the implementation of the SVM algorithms used in HW4 Problems 2 and 3. These consist of: (Problem 2) linear SVM in the primal domain with stochastic sub-gradient descent, (Problem 3) linear and nonlinear SVM in the dual domain. 
 
## Dataset Folders
### bank-note
This dataset is from [UCI repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication). Data descriptions are available in "data-desc.txt". 
## Code
### "HW4.py", "run.sh"
This code is specific to running the tasks specified in HW4 (i.e., learning SVM using the dataset at a variety of hyperparameters and outputting the optimal weight vectors and training/test errors). To run my code and get printed outputs, use the following lines in the terminal while in the directory "SVM": 
```
chmod u+x run.sh 
./run.sh
```
### "SVM.py"
This is the implementation of the Perceptron learning algorithm for the standard, voted, and averaged methods. Using it defines an instance of the class "PCPT" that contains methods that run the Perceptron algorithm and predict the label of an example. Here is a description of an instance of the class "PCPT": 
   
    Attributes
    __________
        
        Parameters
        __________
        X: augmented training dataset (for n data points with m features, 
            X will be nxm array)
        y: training labels 
        epochs: number of maximum epochs
        r: learning rate
        
        Values
        __________
        Xdim: dimension of augented training examples
        numExamples: number of data examples
        w_std: learned weight vector for standard algorithm
        w_vot: learned weight vectors for voted algorithm
        c_vot: counts (# of correctly predicted training examples) for voted 
           algorithm
        a_avg: learned weight vector for averaged algorithm

Running Perceptron on an instance of this class is done using the "std_alg()" method, i.e., "PCPT.std_alg()", for the standard method. The voted and averaged methods are run using the "vot_alg()" and "avg_alg()" methods. When run, this learns the weight vectors that can be accessed by, for example, "PCPT.w_std" for the standard method. 
