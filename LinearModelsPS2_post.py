import numpy as np
from numpy import linalg as la
from tabulate import tabulate

# Performs a pooled OLS regression of y on x. 
# Returns standard errors, t-values, R^2, and the covariance matrix
def estimate(y, x, transform='', T=None, robust_se=False) -> dict:

    assert y.ndim == 2, 'y must be 2D'
    assert x.ndim == 2, 'x must be 2D'
    assert y.shape[1] == 1, 'y must be a column vector'
    assert y.shape[0] == x.shape[0], 'y and x must have the same number of rows'

    # Estimate coefficients using OLS
    b_hat = est_ols(y, x)

    # Compute residuals and R²
    residuals = y - x @ b_hat
    SSR = np.sum(residuals ** 2)
    SST = np.sum((y - y.mean()) ** 2)
    R2 = 1.0 - SSR / SST

    # Compute variance and standard errors
    sigma2, cov, se = variance(transform, SSR, x, T)
    if robust_se:
        cov, se = robust(x, residuals, T)

    # Compute t-values
    t_values = b_hat / se

    # Store results in a dictionary format
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma2, t_values, R2, cov]
    
    return dict(zip(names, results))

def est_ols(y: np.array, x: np.array) -> np.array:
    """
    Estimates beta coefficients using Ordinary Least Squares (OLS).
    """
    return la.inv(x.T @ x) @ (x.T @ y)

# Computes variance, covariance, and standard errors for OLS.
def var_est(trans: str, SSR: float, x: np.array, T: int) -> tuple:

    NT,K = x.shape

    if trans not in {'', 'fd', 'be', 'fe', 're'}:
        raise ValueError(f"Invalid transform: {trans}")

    # Det degrees of freedom 
    if trans in ('', 'fd', 'be', 're'):
        df = K 
    elif trans == 'fe': 
        N = int(NT/T)
        df = N + K 

    sigma2 = SSR / (NT - df) 
    
    cov = sigma2*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma2, cov, se


# Computes the robust variance estimator
def robust( x: np.array, residual: np.array, T:int) -> tuple:

    if (not T) or (T == 1): # cross-sectional robust variance
        Ainv = la.inv(x.T@x)
        uhat2_x = (residual ** 2) * x # element-wise multiplication
        cov = Ainv @ (x.T@uhat2_x) @ Ainv
    
    else: # loop over each individual
        NT,K = x.shape
        N = int(NT / T)
        B = np.zeros((K, K))

        for i in range(N):
            idx_i = slice(i*T, (i+1)*T) # index values
            Omega = residual[idx_i]@residual[idx_i].T # (T,T) matrix, outer product of residuals 
            B += x[idx_i].T @ Omega @ x[idx_i] # (K,K) contribution 

        Ainv = la.inv(x.T @ x)
        cov = Ainv @ B @ Ainv
    
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return cov, se


# Displays regression output in a formatted table
def display_results(labels: tuple, results: dict, 
                    headers=None, title="Output", 
                    _lambda: float = None, **kwargs) -> None:
    
    # Default column headers if not provided
    if headers is None:
        headers = ["", "Coef", "Std. Err", "t-stat"]
    
    # Extract labels
    label_y, label_x = labels
    if not isinstance(label_x, list):
        raise TypeError("Independent variable labels must be in a list.")
    
    if len(label_x) != results['b_hat'].size:
        raise ValueError("Mismatch: Number of labels and estimated parameters must be equal.")
    
    # Construct table with coefficients, standard errors, and t-values
    output_table = [[var, results['b_hat'][i], results['se'][i], results['t_values'][i]] 
                    for i, var in enumerate(label_x)]
    
    # Print the table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(output_table, headers, **kwargs))
    
    # Print additional model statistics 
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma2').item():.3f}")

    # If lambda exists, print it (used in Random Effects models)
    if _lambda: 
        print(f'\u03bb = {_lambda.item():.3f}')


# applies a transformation matrix to a given dataset
def perm( Q_T: np.array, A: np.array) -> np.array:

    #  matrix.
    M,T = Q_T.shape 
    NT,K = A.shape
    N = int(NT/T)

    # initialize output 
    Z = np.empty((M * N, K))
    
    for i in range(N):
        # Select the relevant rows from A and the corresponding rows in Z
        A_block = A[i * T : (i + 1) * T, :]
        Z_block = Q_T @ A_block  # Apply transformation
        
        # Store the transformed data in the correct position in Z
        Z[i * M : (i + 1) * M, :] = Z_block

    return Z


# Applies a transformation matrix to grouped data
def perm_general(Q_T, A, T:int, N:int):
    # Determine the number of rows in the transformed output
    M = Q_T.shape[0] if Q_T.shape[0] != T else T

    # Initialize output matrix
    Z = np.zeros((N * M, A.shape[1]))

    # Apply transformation to each group
    for i in range(N):
        Z[i * M:(i + 1) * M] = Q_T @ A[i * T:(i + 1) * T]

    return Z


def zstex(Z0, n, t):
    k = Z0.shape[1]
    A = Z0.T.reshape(t*k, n, order='F').T
    Z = np.zeros((n*(t - 1), (t - 1)*t*k))
    for i in range(n):
        zi = np.kron(np.eye(t - 1), A[i])
        Z[i*(t - 1): (i + 1)*(t - 1)] = zi
    return Z


def zpred(Z0, n, t):
    k = Z0.shape[1]
    Z = np.zeros((n*(t - 1), int((t - 1)*t*k/2)))
    dt = np.arange(t).reshape(-1, 1)
    
    for i in range(n):
        zi = np.zeros((t - 1, int(t*(t - 1)*k/2)))
        z0i = Z0[i*t: (i + 1)*t - 1]
        
        a = 0
        for j in range(1, t):
            dk = dt[dt < j].reshape(-1, 1)
            b = dk.shape[0]*Z0.shape[1]
            zit = z0i[dk].T.reshape(1, b, order='F')
            zi[j - 1, a: a + b] = zit
            a += b
        Z[i*(t - 1): (i + 1)*(t - 1)] = zi
    return Z


def load_example_data():
    # First, import the data into numpy.
    data = np.loadtxt('wagepan.txt', delimiter=",")
    id_array = np.array(data[:, 0])

    # Count how many persons we have. This returns a tuple with the 
    # unique IDs, and the number of times each person is observed.
    unique_id = np.unique(id_array, return_counts=True)
    n = unique_id[0].size
    t = int(unique_id[1].mean())
    year = np.array(data[:, 1], dtype=int)

    # Load the rest of the data into arrays.
    y = np.array(data[:, 8]).reshape(-1, 1)
    x = np.array(
        [np.ones((y.shape[0])),
            data[:, 2],
            data[:, 4],
            data[:, 6],
            data[:, 3],
            data[:, 9],
            data[:, 5],
            data[:, 7]]
    ).T

    # Lets also make some variable names
    label_y = 'Log wage'
    label_x = [
        'Constant',
        'Black',
        'Hispanic',
        'Education',
        'Experience',
        'Experience sqr',
        'Married',
        'Union'
    ]
    return y, x, n, t, year, label_y, label_x