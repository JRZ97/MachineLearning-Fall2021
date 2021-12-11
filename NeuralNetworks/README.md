# Neural Networks
This is the implementation of neural networks used in HW5 Problem 2. This consists of a three-layer neural network with stochastic gradient descent for classification. 
 
## Dataset Folders
### bank-note
This dataset is from [UCI repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication). Data descriptions are available in "data-desc.txt". 
## Code
### "HW5_NN.py", "run.sh"
This code is specific to running the tasks specified in Problem 2 of HW5 (i.e., learning a neural network using the dataset at a variety of widths and outputting the training/test errors). NOTE: I have commented out the plotting lines I used in tuning the hyperparameters to ensure convergence using stochastic sub-gradient descent at the top of the file. 

To run my code and get printed outputs, use the following lines in the terminal while in the directory "NeuralNetworks": 
```
chmod u+x run.sh 
./run.sh
```
### "NN.py"
This is the implementation of neural networks. Using it defines an instance of the class "NN" that contains methods that run the stochastic gradient descent algorithm and predict the label of an example. Here is a description of an instance of the class "NN": 
        
    Attributes
    __________
        
        Parameters
        __________
        X: augmented training dataset (for n data points with m features, 
            X will be nx(m+1) array)
        y: training labels 
        d: width of NN (number of units in hidden layers 1 and 2)
        weight_initTYpe: "test" for verifying with paper problem 3
                         "normal" for random numbers generated from 
                                  standard Gaussian distribution
        loss: list of squared errors after each weight vector update
        
        Values
        __________
        Xdim: dimension of augented training examples
        numExamples: number of data examples

Running stochastic gradient descent is done using the sgd method. Here is a description of the method: 
        
        Parameters
        ----------
        epochs: number of maximum epochs
        gamma_0, a: hyper-parameters in learning rate, 
            gamma_t = gamma_0 / (1 + gamma_0*t/a)
        
        Returns
        -------
        self.w1 = dx(k+1) array of weight values of the first layer, where k 
            is the number of features in the input datset X,
            [[w01, w11, w21, ..., wk1]
             [w02, w12, w22, ..., wk2]
             ...
             w0d, w1d, w2d, ..., wkd]
        self.w2 = dx(d+1) array of weight values of the second layer, 
            [[w01, w11, w21, ..., wd1]
             [w02, w12, w22, ..., wd2]
             ...
             w0d, w1d, w2d, ..., wdd]
        self.w3 = d+1, array of weight values of the third layer, 
            [w01, w11, w21, ..., wd1]
            
Output predictions are made using the pred method. 
