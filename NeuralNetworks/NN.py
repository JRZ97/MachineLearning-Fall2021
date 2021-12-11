"""
This will be my implementation of neural networks for HW5.
"""

import numpy as np

class NN:
    """
    Contains all the information involved in creating a 3 layer neural 
    network.
    
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
    """
    def __init__(self, X, y, d, weight_initType):
        self.X = X
        self.y = y
        self.d = d - 1 # don't include the bias nodes in width
        self.Xdim = np.size(X,1)
        self.numExamples = np.size(y)
        self.loss = list()
        self.weight_initType = weight_initType
        
        # initialize weights 
        self._weight_init()
        
    def sgd(self, epochs, gamma_0, a):
        """
        Stochastic gradient descent.

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
        """
        epoch_order = np.arange(self.numExamples)
        t = 0 # initialize number of iterations (epochs)
        
        for epoch in range(epochs):
            # shuffle the data
            np.random.shuffle(epoch_order)
            t += 1 # increment number of iterations (epochs)
            
            # for each training example 
            for i in epoch_order:
                x = self.X[i]
                y = self.y[i]
                
                # calculate gradient of loss
                self._fwd_prop(x,y)
                self._back_prop(x)
                                
                # calculate learning rate
                gamma_t = gamma_0 / (1 + gamma_0*t/a)
                
                # update weight vectors
                self.w1 -= gamma_t * self.dw1 
                self.w2 -= gamma_t * self.dw2
                self.w3 -= gamma_t * self.dw3 
    
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
            self.z1 = np.insert(np.array([self._sigmoid(self.w1[i,:].dot(x))\
                                          for i in range(self.d)]),0,1)
            self.z2 = np.insert(np.array([self._sigmoid(self.w2[i,:].dot(self.z1))\
                                          for i in range(self.d)]),0,1)
            y_fwdProp = self.w3.dot(self.z2)
            if y_fwdProp < 0:
                y_pred[i] = -1
        return y_pred 
        
    def _fwd_prop(self, x, y_star):
        """
        Forward propogation.

        Parameters
        ----------
        x: single data example
        y_star: label for x
        
        Returns
        -------
        self.z1: 1x(d+1) array of node values of the first layer, 
            [z_0^1, z_1^1, z_2^1, ..., z_d^1]
        self.z2: 1x(d+1) array of node values of the second layer, 
            [z_0^2, z_1^2, z_2^2, ..., z_d^2]
        self.dL_dy: y - y_star
        self.loss: loss (objective) function evaluation
        """
        self.z1 = np.insert(np.array([self._sigmoid(self.w1[i,:].dot(x))\
                                      for i in range(self.d)]),0,1)
        self.z2 = np.insert(np.array([self._sigmoid(self.w2[i,:].dot(self.z1))\
                                      for i in range(self.d)]),0,1)
        y = self.w3.dot(self.z2)
        self.dL_dy = y - y_star
        self.loss.append(0.5 * (y - y_star)**2)
        
        if self.weight_initType == "test":
            with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.5f}'.format}, linewidth=100):
                print(f"\t For Layer 1, [z1, z2] = {self.z1[1:]}.")
                print(f"\t For Layer 2, [z1, z2] = {self.z2[1:]}.")
                print(f"\t dL_dy = {self.dL_dy:0.4f}.")
        
    def _back_prop(self, x):
        """
        Backward propogation.
        
        Parameters
        ----------
        x: single data example

        Returns
        -------
        self.dw1 = dx(k+1) array of partial derivatives of the loss w.r.t.
            weight values of the first layer, where k is the number of 
            features in the input datset X,
            [[dL_dw01, dL_dw11, dL_dw21, ..., dL_dwk1]
             [dL_dw02, dL_dw12, dL_dw22, ..., dL_dwk2]
             ...
             dL_dw0d, dL_dw1d, dL_dw2d, ..., dL_dwkd]
        self.dw2 = dx(d+1) array of partial derivatives of the loss w.r.t.
            weight values of the second layer, 
            [[dL_dw01, dL_dw11, dL_dw21, ..., dL_dwd1]
             [dL_dw02, dL_dw12, dL_dw22, ..., dL_dwd2]
             ...
             dL_dw0d, dL_dw1d, dL_dw2d, ..., dL_dwdd]
        self.dw3 = d+1, array of partial derivatives of the loss w.r.t.
            weight values of the third layer, 
            [dL_dw01, dL_dw11, dL_dw21, ..., dL_dwd1]
        
        """
        # step 1
        self.dw3 = self.dL_dy * self.z2
        
        # step 2
        self.dz2 = self.dL_dy * self.w3[1:]
        
        theta = self.dz2 * self.z2[1:] * (1-self.z2[1:])
        self.dw2 = np.outer(theta,self.z1)
        
        # step 3
        self.dz1 = self.w2[:,1:].T.dot(theta)
        
        phi = self.dz1 * self.z1[1:] * (1-self.z1[1:])
        self.dw1 = np.outer(phi,x)

        if self.weight_initType == "test":
            with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.5f}'.format}, linewidth=100):
                print(f"\t For Layer 1, loss derivative w.r.t. [w01, w11, w21] = {self.dw1[0,:]}.")
                print(f"\t For Layer 1, loss derivative w.r.t. [w02, w12, w22] = {self.dw1[1,:]}.")
                print(f"\t For Layer 2, loss derivative w.r.t. [w01, w11, w21] = {self.dw2[0,:]}.")
                print(f"\t For Layer 2, loss derivative w.r.t. [w02, w12, w22] = {self.dw2[1,:]}.")
                print(f"\t For Layer 3, loss derivative w.r.t. [w01, w11, w21] = {self.dw3}.")
    
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

    def _weight_init(self):
        """
        Initializes the weight values according to the user input. 

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
        """
        if self.weight_initType == "test":
            # test weights (Table 1 in HW5)
            self.w1 = np.array([[-1,-2,-3],
                                [1,2,3]])
            self.w2 = np.array([[-1,-2,-3],
                                [1,2,3]])
            self.w3 = np.array([-1,2,-1.5])
        elif self.weight_initType == "normal":
            self.w1 = np.random.normal(size = (self.d,self.Xdim))
            self.w2 = np.random.normal(size = (self.d,self.d+1))
            self.w3 = np.random.normal(size = self.d+1)
        elif self.weight_initType == "zeros":
            self.w1 = np.zeros((self.d,self.Xdim))
            self.w2 = np.zeros((self.d,self.d+1))
            self.w3 = np.zeros(self.d+1)
    
    
    
    
    
    
    
    
    