"""
This will be my implementation of SVM for HW4.
"""

import numpy as np
import scipy.optimize
from scipy.spatial.distance import cdist

class SVM:
    """
    Contains all the information involved in using the SVM algorithm.
    
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
        
    def prim_alg(self, epochs, C, eval_J = 1, gamma_sched = '2a', gamma_0 = 1, a = 1):
        """
        Runs the primal domain SGD algorithm. 
        
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
        """
        # initialize weight vector
        self.w_prim = np.zeros(self.Xdim)
        
        # initialize J 
        J = []
        
        # run SGD
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
                
                w0_aug = np.concatenate([self.w_prim[:-1],np.array([0])])
                # calculate sub-gradient
                if y * np.dot(self.w_prim, x) <= 1:
                    del_J = w0_aug - C * self.numExamples * y * x
                else:
                    del_J = w0_aug
                                
                # calculate learning rate
                if gamma_sched == '2a':
                    gamma_t = gamma_0 / (1 + gamma_0*t/a)
                elif gamma_sched == '2b':
                    gamma_t = gamma_0 / (1 + t)
                else:
                    raise ValueError('This type of schedule for the learning rate is not currently implemented!')
                
                # update w
                self.w_prim -= gamma_t * del_J
                
                # evaluate SVM objective function to observe convergence
                if eval_J == 1:
                    J.append(0.5 * np.dot(w0_aug, w0_aug) \
                        + C * sum([max(0, 1 - self.y[i]*np.dot(self.w_prim, self.X[i])) \
                                   for i in range(self.numExamples)]))
        
        if eval_J == 1:
            return self.w_prim, J
        elif eval_J == 0:
            return self.w_prim
        else: 
            raise ValueError('Not valid indicator for evaluating the objective function!')
    
    def prim_pred(self, X):
        """
        Predicts using the learned weight vector from the primal form SGD 
        algorithm. 
        
        Parameters
        __________
        X: augmented dataset
                  
        Returns
        __________
        y_pred: label predictions
        """
        y_pred = np.ones(np.size(X,0))
        
        for i,x in enumerate(X):
            if np.dot(self.w_prim,x) < 0:
                y_pred[i] = -1
        return y_pred    
    
    def dual_alg(self, C):
        """
        Runs the linear dual domain algorithm. 
        
        Parameters
        __________
        C: hyperparameter for the tradeoff between the regularizer and 
            the hinge loss
            
        Returns
        __________
        self.w_dual: learned weight vectors for dual domain algorithm
        """    
        # initialize weight vector
        self.w_dual = np.zeros(self.Xdim)

        # define the bounds of Lagrangian multipliers alpha
        bnds = scipy.optimize.Bounds(0,C)
        
        # load constraints into dictionary
        cons = ({'type':'eq','fun': self._constraint}) # enforce = 0
        
        # initialize the Lagrangian multipliers alpha
        alpha0 = np.zeros(self.numExamples)
        
        # run optimization to find optimal Lagrangian multipliers alpha_star
        sol = scipy.optimize.minimize(fun = self._objective, \
                        x0 = alpha0, method = 'SLSQP', constraints = cons, \
                        bounds = bnds)
        alpha_star = sol.x
        
        X = self.X[:,:-1] # train on non-augmented X!
        y = self.y
        
        # calculate optimal (non-augmented) weight vector w0_star
        w0_star = 0
        for i in range(self.numExamples):
            w0_star += alpha_star[i]*y[i]*X[i]
        
        # find optimal bias parameter b_star
        thresh = 1e-6 # threshold on whether 0 < alpha_star < C
        b_starArr = [y[j] - np.dot(w0_star,X[j]) for j in \
                      range(self.numExamples) if alpha_star[j]>thresh \
                      and alpha_star[j]<C-thresh]
        b_star = sum(b_starArr) / len(b_starArr)
        self.w_dual = np.concatenate([w0_star,np.array([b_star])])
        return self.w_dual
        
    def _objective(self, alpha):
        """
        Objective function to mimimize in solving the linear dual SVM form. 
    
        Parameters
        ----------
        alpha: (n,) array of Lagrangian multipliers alpha, where n is the 
            number of training examples
    
        Returns
        -------
        J: objective function evaluation
        """
        X = self.X[:,:-1] # train on non-augmented X!
        y = self.y
        
        # calculate J using matrix multiplication
        K = X @ X.T
        int_mat = 0.5 * np.outer(y,y) * K * np.outer(alpha,alpha)
        J = np.sum(int_mat) - sum(alpha)
        return J
                    
    def _constraint(self, alpha):
        """
        Equality constraint in solving the linear dual SVM form. 

        Parameters
        ----------
        alpha: (n,) array of Lagrangian multipliers alpha, where n is the 
            number of training examples

        Returns
        -------
        Constraint evaluation, sum_i (alpha_i * y_i) = 0 
        """
        return np.dot(alpha,self.y) 

    def dual_pred(self, X):
        """
        Predicts using the learned weight vector from the linear dual form 
        algorithm. 
        
        Parameters
        __________
        X: augmented dataset
                  
        Returns
        __________
        y_pred: label predictions
        """
        y_pred = np.ones(np.size(X,0))
        
        for i,x in enumerate(X):
            if np.dot(self.w_dual,x) < 0:
                y_pred[i] = -1
        return y_pred  
    
    def dual_NLalg(self, C, kern = 'Gauss', gamma = None):
        """
        Runs the nonlinear dual domain algorithm. 
        
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
        """    
        # save kernal indicator
        self.kern = kern
        if gamma:
            self.gamma = gamma
        
        # save hyperparameter C
        self.C_NL = C
        
        # initialize weight vector
        self.NL_alphaStar = np.zeros(self.numExamples)

        # define the bounds of Lagrangian multipliers alpha
        bnds = scipy.optimize.Bounds(0,C)
        
        # load constraints into dictionary
        cons = ({'type':'eq','fun': self._constraint}) # enforce = 0
        
        # initialize the Lagrangian multipliers alpha
        alpha0 = np.zeros(self.numExamples)
        
        # run optimization to find optimal Lagrangian multipliers alpha_star
        sol = scipy.optimize.minimize(fun = self._NLobjective, \
                        x0 = alpha0, method = 'SLSQP', constraints = cons, \
                        bounds = bnds)
        self.NL_alphaStar = sol.x
        return self.NL_alphaStar

        
    def _NLobjective(self, alpha):
        """
        Objective function to mimimize in solving the nonlinear dual SVM form. 
    
        Parameters
        ----------
        alpha: (n,) array of Lagrangian multipliers alpha, where n is the 
            number of training examples
    
        Returns
        -------
        J: objective function evaluation
        """
        X = self.X[:,:-1] # train on non-augmented X!
        y = self.y
        
        # calculate J using matrix multiplication
        K = self._NLkernel(X,X)
        int_mat = np.outer(y,y) * K * np.outer(alpha,alpha)
        J = 0.5 * np.sum(int_mat) - sum(alpha)
        return J
                    
    def _NLkernel(self, A, B):
        """
        Kernel evaluation for nonlinear dual SVM form. 
        possibilities: 'Gauss' where
                'Gauss' is the Gaussian kernel 

        Parameters
        ----------
        A, B: matrices/vectors to compute the Kernel over

        Returns
        -------
        Kernel evaluation
        """
        if self.kern == 'Gauss': 
            pairwise_sq_dists = cdist(A, B, 'sqeuclidean')
            return np.exp(-pairwise_sq_dists/self.gamma)
        else:
            raise ValueError('This type of kernel is not currently implemented!')
        
    def dual_NLpred(self, X):
        """
        Predicts using the learned weight vector from the nonlinear dual form 
        algorithm. 
        
        Parameters
        __________
        X: augmented dataset
                  
        Returns
        __________
        y_pred: label predictions
        """
        y_pred = np.ones(np.size(X,0))
        
        # calculate b
        thresh = 1e-6 # threshold on whether 0 < alpha_star < C
        b_starArr = [self.y[j] - np.dot(self.NL_alphaStar * self.y,\
                self._NLkernel(self.X[:,:-1],self.X[j,:-1].reshape(1,-1)))\
                for j in range(self.numExamples) if self.NL_alphaStar[j]\
                >thresh and self.NL_alphaStar[j]<self.C_NL-thresh]
        b_star = sum(b_starArr) / len(b_starArr)
        for i,x in enumerate(X):
            # calculate dot(w^*,phi(x))
            wStar_dot_phi = np.dot(self.NL_alphaStar * self.y,\
                            self._NLkernel(self.X[:,:-1],x[:-1].reshape(1,-1)))
                        
            if wStar_dot_phi + b_star < 0:
                y_pred[i] = -1
        return y_pred     
    
    
    
    
    
    
    
    
    