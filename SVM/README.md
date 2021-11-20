# SVM
This is the implementation of the SVM algorithms used in HW4 Problems 2 and 3. These consist of: (Problem 2) linear SVM in the primal domain with stochastic sub-gradient descent, (Problem 3) linear and nonlinear SVM in the dual domain. 
 
## Dataset Folders
### bank-note
This dataset is from [UCI repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication). Data descriptions are available in "data-desc.txt". 
## Code
### "HW4.py", "run.sh"
This code is specific to running the tasks specified in HW4 (i.e., learning SVM using the dataset at a variety of hyperparameters and outputting the optimal weight vectors and training/test errors). NOTE: I have commented out the plotting lines I used in tuning the hyperparameters to ensure convergence using stochastic sub-gradient descent in Problem 2. 

To run my code and get printed outputs, use the following lines in the terminal while in the directory "SVM": 
```
chmod u+x run.sh 
./run.sh
```
### "SVM.py"
This is the implementation of the SVM learning algorithm in the primal and dual domains. Using it defines an instance of the class "SVM" that contains methods that run the SVM algorithm and predict the label of an example. Here is a description of an instance of the class "SVM": 
        
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

Running the primal form of SVM is done using the prim_alg method. Here is a description of the method: 
        
        Parameters
        __________
        epochs: number of maximum epochs
        C: hyperparameter for the tradeoff between the regularizer and 
            the hinge loss
        eval_J: whether or not to evaluate and return the objective function; 
            default = 1
            possibilities: 1 (True) and 0 (False)
        gamma_sched: schedule of learning rate; default = '2a'
            possibilities: '2b' and '2c' where, with t iterations (epochs),
                '2a' has gamma_t = gamma_0/(1+gamma_0*t/a)
                '2b' has gamma_t = gamma_0/(1+t)
            (uses inputs gamma_0; default = 1 and a; default = 1)
            
        Returns 
        __________
        self.w_prim: learned weight vectors for primal domain SGD algorithm
        J: SVM objective function evaluation after each weight vector update

Running the linear dual form of SVM is done using the dual_alg method. Here is a description of the method: 
        
        Parameters
        __________
        C: hyperparameter for the tradeoff between the regularizer and 
            the hinge loss
            
        Returns
        __________
        self.w_dual: learned weight vectors for dual domain algorithm
        
Running the nonlinear dual form of SVM is done using the dual_NLalg method. Here is a description of the method: 
        
        Parameters
        __________
        C: hyperparameter for the tradeoff between the regularizer and 
            the hinge loss
        kern: indication of which type of kernel to use; default = 'Gauss'
            possibilities: 'Gauss' where
                'Gauss' is the Gaussian kernel 
                    parameters: gamma; default = None
        Returns
        __________
        self.NL_alphaStar: optimal Lagrangian multipliers for nonlinear
            dual domain algorithm
            
Output predictions are made using the prim_pred, dual_pred, dual_NLpred methods for the linear primal, linear dual, and nonlinear dual forms of SVM, respectively. 
