# Perceptron
This is the implementation of the Perceptron algorithm used in HW3 Problem 2. These consist of: part (a) the standard Perceptron algorithm, part (b) the voted Perceptron algorithm, part (c) the averaged Perceptron algorithm.
 
## Dataset Folders
### bank-note
This dataset is from [UCI repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication). Data descriptions are available in "data-desc.txt". 
## Code
### "HW3.py", "run.sh"
This code is specific to running the tasks specified in HW3 (i.e., running Perceptron using a maximum number of epochs of 10 and a learning rate of 0.01 and printing out the learned weight vectors and average prediction error on the test dataset). To run my code and get printed outputs, use the following lines in the terminal while in the directory "Perceptron": 
```
chmod u+x run.sh 
./run.sh
```
### "perceptron.py"
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
