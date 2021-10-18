import numpy as np

def analyt(X,y):
    """
    Analytical solution to LMS regression. 
    
    Parameters
    ----------
    X : Input vector. dxm numpy array, where there are m number of 
        d-dimensional vectors. We assume that x_1 is always 1 after feature 
        augmentation for each vector (meaning that the first row of X is ones)
    y : Output vector. mx1 numpy array for m outputs. 

    Returns
    -------
    w : Weight vector. dx1 numpy array, where the first element w_0 is the 
        bias parameter. 

    """
    piece_1 = np.linalg.inv(np.matmul(X, np.transpose(X))) 
    piece_2 = np.matmul(X,y)
    return np.matmul(piece_1, piece_2)

def BGD(X,y,r,thresh):
    """
    Batch gradient descent algorithm for LMS regression. 
    
    Parameters
    ----------
    X : Input vector. dxm numpy array, where there are m number of 
        d-dimensional vectors. We assume that x_1 is always 1 after feature 
        augmentation for each vector (meaning that the first row of X is ones)
    y : Output vector. mx1 numpy array for m outputs. 
    r : Learning rate. 
    thresh : Error threshold. BGD will only exit when the norm of the weight 
             vector difference between steps is less than this tolerance 
             threshold. DEFAULT: np.inf, for testing purposes, where it runs 
             a fixed 1000 times. 

    Returns
    -------
    w : Weight vector. dx1 numpy array, where the first element w_0 is the 
        bias parameter.
    J : Value of the cost function evaluated at every step. This is the 
        sum of squared costs over the inputs. 

    """
    num_dims = np.shape(X)[0] # number of dimensions
    num_examples = np.shape(X)[1] # number of examples
    w = np.zeros(num_dims) # initialize weight vector
    
    weight_relErr = 1 # initialize error 
    J = list() # initialize list of cost function evaluations
    
    if thresh == np.inf: 
        for test in range(1000): # run this for testing for convergence
            err_arr = np.array([y[i] - np.matmul(w, X[:,i]) for i in range(num_examples)])
            J.append(0.5 * sum(np.square(err_arr)))
            
            # evaluate gradient of J
            J_grad = np.zeros(num_dims)
            
            # try the component-wise way (ENDED UP BEING SLOWER WAY)
            # for j in range(num_dims):
            #     J_grad[j] = -sum([err_arr[i]*X[j,i] for i in range(num_examples)])
            
            # # try the matrix multiplication way (ENDED UP BEING FASTER WAY)
            J_grad = -np.sum(err_arr * X,1)
            
            w_update = w - r * J_grad
            weight_relErr = np.linalg.norm(w_update - w)
            w = w_update
    else:
        while weight_relErr > thresh: 
            err_arr = np.array([y[i] - np.matmul(w, X[:,i]) for i in range(num_examples)])
            J.append(0.5 * sum(np.square(err_arr)))
            
            # evaluate gradient of J
            J_grad = np.zeros(num_dims)
            
            # try the component-wise way (ENDED UP BEING SLOWER WAY)
            # for j in range(num_dims):
            #     J_grad[j] = -sum([err_arr[i]*X[j,i] for i in range(num_examples)])
            
            # # try the matrix multiplication way (ENDED UP BEING FASTER WAY)
            J_grad = -np.sum(err_arr * X,1)
            
            w_update = w - r * J_grad
            weight_relErr = np.linalg.norm(w_update - w)
            w = w_update
    
    return w, J
            
def SGD(X,y,r,thresh):
    """
    Stochastic gradient descent algorithm for LMS regression. 
    
    Parameters
    ----------
    X : Input vector. dxm numpy array, where there are m number of 
        d-dimensional vectors. We assume that x_1 is always 1 after feature 
        augmentation for each vector (meaning that the first row of X is ones)
    y : Output vector. mx1 numpy array for m outputs. 
    r : Learning rate. 
    thresh : Error threshold. SGD will only exit when the norm of the cost 
             evaluation difference between steps is less than this tolerance 
             threshold. DEFAULT: np.inf, for testing purposes, where it runs 
             a fixed 1000 times. 

    Returns
    -------
    w : Weight vector. dx1 numpy array, where the first element w_0 is the 
        bias parameter.
    J : Value of the cost function evaluated at every step. This is the 
        sum of squared costs over the inputs. 

    """
    num_dims = np.shape(X)[0] # number of dimensions
    num_examples = np.shape(X)[1] # number of examples
    w = np.zeros(num_dims) # initialize weight vector
    
    cost_relErr = 1 # initialize error 
    J = list() # initialize list of cost function evaluations
    J.append(1e6) # initialize, will be popped out at the end
    
    if thresh == np.inf: 
        for test in range(10000): # run this for testing for convergence
            rand_ind = np.random.randint(num_examples)   
            err = y[rand_ind] - np.matmul(w, X[:,rand_ind])
            w += r * err * X[:,rand_ind]
            
            err_arr = np.array([y[i] - np.matmul(w, X[:,i]) for i in range(num_examples)])
            J.append(0.5 * sum(np.square(err_arr)))
            cost_relErr = np.linalg.norm(J[-1] - J[-2])
    else:
        while cost_relErr > thresh: 
            rand_ind = np.random.randint(num_examples)      
            err = y[rand_ind] - np.matmul(w, X[:,rand_ind])
            w += r * err * X[:,rand_ind]
            
            err_arr = np.array([y[i] - np.matmul(w, X[:,i]) for i in range(num_examples)])
            J.append(0.5 * sum(np.square(err_arr)))
            cost_relErr = np.linalg.norm(J[-1] - J[-2])
    
    J.pop(0) # pop out that first initialized element
    return w, J
    
    
    
    
    
    
    
    