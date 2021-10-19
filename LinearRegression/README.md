# Linear Regression
These are LMS linear regression learning algorithms used in HW2. We used it to implement the batch gradient descent (BGD) algorithm, stochastic gradient descent (SGD) algorithm, and the analytical solutiion to the LMS linear regression problem. 
## Dataset Folders
### concrete
This dataset is from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test). Data descriptions are available in "data-desc.txt". 
## Code
### "regression.py", "run.sh"
This code is specific to running the tasks specified in HW1 (i.e., reporting in tables the avrage prediction errors on the Car/Bank datasets using different measures of purity. To run my code and get printed outputs, use the following lines in the terminal while in the directory "LinearRegression": 
```
chmod u+x run.sh 
./run.sh
```
Plotting lines are commented out (lines 39 to 45 and lines 65 to 71) if you would like to verify the plotting of my cost function values. 
### "LMS.py"
These is the implementations of the LMS linear regression algorithms. These are the following descriptions of the functions. 
```
analyt(X,y):
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
```
```
BGD(X,y,r,thresh):
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
```
```
SGD(X,y,r,thresh):
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
```
