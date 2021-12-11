"""
This will be my implementation of logistic regression for HW5.
"""

import numpy as np

class LOGSTC:
    """
    Contains all the information involved in implementing a logistic 
    regression model with stochastic gradient descent.  
    
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
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.Xdim = np.size(X,1)
        self.numExamples = np.size(y)
        self.obj = list()
        
    def sgd(self, epochs, gamma_0, a, v, train_type, conv_test = "F"):
        """
        Stochastic gradient descent.

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
        """
        epoch_order = np.arange(self.numExamples)
        t = 0 # initialize number of iterations (epochs)
        self.w = np.zeros(self.Xdim)
        
        for epoch in range(epochs):
            # shuffle the data
            np.random.shuffle(epoch_order)
            t += 1 # increment number of iterations (epochs)
            
            # for each training example 
            for i in epoch_order:
                x = self.X[i]
                y = self.y[i]
                
                # calculate gradient of loss
                s = y * self.w.dot(x)
                if train_type == "MLE":
                    del_L = -self.numExamples * (1 - self._sigmoid(s)) \
                        * y * x
                elif train_type == "MAP":
                    del_L = -self.numExamples * (1 - self._sigmoid(s)) \
                        * y * x + self.w/v
                        
                if conv_test == "T": 
                    if train_type == "MLE":
                        self.obj.append(self.numExamples *  np.log(1 + \
                                        np.exp(-y * self.w.dot(x))))
                    elif train_type == "MAP":
                        self.obj.append(self.numExamples *  np.log(1 + \
                                        np.exp(-y * self.w.dot(x))) + \
                                        self.w.dot(self.w)/(2*v))
                
                # calculate learning rate
                gamma_t = gamma_0 / (1 + gamma_0*t/a)
                
                # update weight vector
                self.w -= gamma_t * del_L
              
    def pred(self, X):
        """
        Predicts using the learned weight vector. 
        
        Parameters
        ----------
        X: augmented dataset
                  
        Returns
        ----------
        y_pred: label predictions
        """
        y_pred = np.ones(np.size(X,0))
        
        for i,x in enumerate(X):
            prob_calc = self._sigmoid(self.w.dot(x))
            
            if prob_calc < 0.5:
                y_pred[i] = -1
        return y_pred 
    
    def _sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters
        ----------
        x: single data example

        Returns
        -------
        Sigmoid function evaluation. 
        """
        return 1 / (1 + np.exp(-x))
                
                
                
                
                
                
                
                
                
                