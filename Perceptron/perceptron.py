"""
This will be my implementation of Perceptron for HW3.
"""

import numpy as np

class PCPT:
    """Contains all the information involved in using the Perceptron
    algorithm.
    
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
    """
    
    def __init__(self, X, y, epochs, r):
        self.X = X
        self.y = y
        self.epochs = epochs
        self.r = r
        self.Xdim = np.size(X,1)
        self.numExamples = np.size(y)
        
        # standard algorithm
        self.w_std = np.zeros(self.Xdim)
        
        # voted algorithm
        self.w_vot = list()
        self.c_vot = list()
        
        # averaged algorithm
        self.a_avg = np.zeros(self.Xdim)
        
    def std_alg(self):
        """Runs the standard Perceptron algorithm. 
        
        Parameters
        __________
        None
                  
        Return 
        __________
        self.w_std: learned weight vector for standard algorithm 
        """
        self.w_std = np.zeros(self.Xdim)
        epoch_order = np.arange(self.numExamples)
        for epoch in range(self.epochs):
            # shuffle the data
            np.random.shuffle(epoch_order)
            
            # for each training example 
            for i in epoch_order:
                x = self.X[i]
                # if there is a prediction error, update the weight vector
                if self.y[i] * np.dot(self.w_std, x) <= 0:
                    self.w_std += self.r * self.y[i] * x                
        return self.w_std
    
    def vot_alg(self):
        """Runs the voted Perceptron algorithm. 
        
        Parameters
        __________
        None
                  
        Return 
        __________
        self.w_vot: learned weight vectors for voted algorithm 
        self.c_vot: counts (# of correctly predicted training examples) for 
           voted algorithm
        """
        self.w_vot = list()
        self.c_vot = list()
        epoch_order = np.arange(self.numExamples)
        w_curr = np.zeros(self.Xdim)
        c = 0
        for epoch in range(self.epochs):
            # shuffle the data
            np.random.shuffle(epoch_order)
            
            # for each training example 
            for i in epoch_order:
                x = self.X[i]
                # if there is a prediction error, update current weight 
                # vector, set count equal to 1
                if self.y[i] * np.dot(w_curr, x) <= 0:
                    self.w_vot.append(w_curr.copy())
                    self.c_vot.append(c)
                    w_curr += self.r * self.y[i] * x
                    c = 1
                # else, iterate the current count 
                else:
                    c += 1
            self.w_vot.pop(0)
            self.c_vot.pop(0)
        return self.w_vot, self.c_vot

    def avg_alg(self):
        """Runs the averaged Perceptron algorithm. 
        
        Parameters
        __________
        None
                  
        Return 
        __________
        self.a_avg: learned weight vector for averaged algorithm 
        """
        self.a_avg = np.zeros(self.Xdim)
        epoch_order = np.arange(self.numExamples)
        w_curr = np.zeros(self.Xdim)
        for epoch in range(self.epochs):
            # shuffle the data
            np.random.shuffle(epoch_order)
            
            # for each training example 
            for i in epoch_order:
                x = self.X[i]
                # if there is a prediction error, update the weight vector
                if self.y[i] * np.dot(w_curr, x) <= 0:
                    w_curr += self.r * self.y[i] * x
                self.a_avg += w_curr.copy()
        return self.a_avg
        
    def std_pred(self, X):
        """Predicts using the learned weight vector from the standard 
        algorithm. 
        
        Parameters
        __________
        X: augmented test dataset
                  
        Return 
        __________
        y_pred: label predictions
        """
        # see if standard algorithm has been run yet
        if np.all(self.w_std == 0): 
            print("Make sure to run algorithm before predicting!")
        
        y_pred = np.ones(np.size(X,0))
        
        for i,x in enumerate(X):
            if np.dot(self.w_std,x) < 0:
                y_pred[i] = -1
        
        return y_pred
    
    def vot_pred(self, X):
        """Predicts using the learned weight vector from the voted 
        algorithm. 
        
        Parameters
        __________
        X: augmented test dataset
                  
        Return 
        __________
        y_pred: label predictions
        """
        # see if voted algorithm has been run yet
        if not self.w_vot: 
            print("Make sure to run algorithm before predicting!")
        
        y_pred = np.ones(np.size(X,0))
        
        for i,x in enumerate(X):
            
            # calculate inner sum
            res = 0
            for j in range(np.size(self.c_vot)):
                if np.dot(self.w_vot[j],x) >= 0:
                    res += self.c_vot[j]
                else:
                    res -= self.c_vot[j]
              
            # outer sign
            if res < 0:
                y_pred[i] = -1
        
        return y_pred
        
    def avg_pred(self, X):
        """Predicts using the learned weight vector from the averaged 
        algorithm. 
        
        Parameters
        __________
        X: augmented test dataset
                  
        Return 
        __________
        y_pred: label predictions
        """
        # see if standard algorithm has been run yet
        if np.all(self.a_avg == 0): 
            print("Make sure to run algorithm before predicting!")
        
        y_pred = np.ones(np.size(X,0))
        
        for i,x in enumerate(X):
            if np.dot(self.a_avg,x) < 0:
                y_pred[i] = -1
        
        return y_pred        
        
        
        
        
        
        
        