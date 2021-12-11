# Logistic Regression
This is the implementation of logistic regression used in HW5 Problem 2. This consists of maximum a priori and maximum likelihood estimation with stocastic gradient descent for classification. 
 
## Dataset Folders
### bank-note
This dataset is from [UCI repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication). Data descriptions are available in "data-desc.txt". 
## Code
### "HW5_LR.py", "run.sh"
This code is specific to running the tasks specified in Problem 3 of HW5 (i.e., learning using logistic regression at a variety of variances and outputting the training/test errors). NOTE: I have commented out the plotting lines I used in tuning the hyperparameters to ensure convergence using stochastic sub-gradient descent at the top of the file. 

To run my code and get printed outputs, use the following lines in the terminal while in the directory "LogisticRegression": 
```
chmod u+x run.sh 
./run.sh
```
### "logistic.py"
This is the implementation of logistic regression. Using it defines an instance of the class "LOGSTC" that contains methods that run the stochastic gradient descent algorithm and predict the label of an example. Here is a description of an instance of the class "LOGSTC": 
        
    Attributes
    __________
        
        Parameters
        __________
        X: augmented training dataset (for n data points with m features, 
            X will be nx(m+1) array)
        y: training labels 
        
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
        v: prior variance
        train_type: "MLE" for maximum likelihood estimation
                    "MAP" for maximum a posteriori estimation
         
        conv_test: "T" if wanting to check convergence of the objective function
                   "F" if not wanting to, default = "F"
        
        Returns
        -------
        self.w: weight vector
            
Output predictions are made using the pred method. 
