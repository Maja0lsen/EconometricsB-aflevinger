import numpy as np
from numpy import linalg as la
from tabulate import tabulate




def estimate(
        y: np.ndarray, x: np.ndarray, z: np.ndarray = None, transform='', N=None, T=None, robust_se=False
    ) -> dict:
    """Computes OLS or IV estimates and returns key statistics."""
    
    # Ensure correct input dimensions
    assert y.ndim == 2 and x.ndim == 2, 'Inputs y and x must be 2D arrays'
    assert y.shape[1] == 1, 'y must be a column vector'
    assert y.shape[0] == x.shape[0], 'y and x must have the same number of observations'
    
    # Determine if instrumental variables are used
    is_iv = z is not None
    
    # Estimate coefficients using OLS or IV
    if is_iv:
        b_hat = est_piv(y, x, z)
    else:
        b_hat = est_ols(y, x)
    
    # Compute residuals and sum of squares
    resid = y - x @ b_hat
    SSR = resid.T @ resid
    SST = (y - y.mean()).T @ (y - y.mean())
    R2 = 1 - SSR / SST if not is_iv else np.nan
    
    # Adjust x for IV estimation
    x_transformed = z @ solve(z.T @ z, z.T @ x) if is_iv else x
    
    # Compute variance, covariance matrix, and standard errors
    sigma, cov, se = variance(transform, SSR, x_transformed, N, T)
    if robust_se:
        cov, se = robust(x_transformed, resid, T)
    
    t_values = b_hat / se
    
    return {'b_hat': b_hat, 'se': se, 'sigma': sigma, 't_values': t_values, 'R2': R2, 'cov': cov}


"""
# Dette er fra øvelsestimen (inspo til ovenstående)
def estimate( 
        y: np.ndarray, x: np.ndarray, transform='', N=None, T=None
    ) -> list:
    
    b_hat = est_ols(y, x)
    resid = y - x@b_hat
    u_hat = resid@resid.T
    SSR = resid.T@resid
    SST = (y - np.mean(y)).T@(y - np.mean(y))
    R2 = 1 - SSR/SST

    sigma, cov, se = variance(transform, SSR, x, N, T)
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma, t_values, R2, cov]
    return dict(zip(names, results))
"""
    
def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return la.inv(x.T@x)@(x.T@y)

def variance( 
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
        N: int,
        T: int
    ) -> tuple :
    """Use SSR and x array to calculate different variation of the variance.

    Args:
        transform (str): Specifiec if the data is transformed in any way.
        SSR (float): SSR
        x (np.ndarray): Array of independent variables.
        N (int, optional): Number of observations. If panel, then the 
        number of individuals. Defaults to None.
        T (int, optional): If panel, then the number of periods an 
        individual is observerd. Defaults to None.

    Raises:
        Exception: [description]

    Returns:
        tuple: [description]
    """

    K=x.shape[1]

    if transform in ('', 're' 'fd'):
          sigma = SSR/(N*T-K)
    elif transform.lower() == 'fe':
          sigma = SSR/(N*T-N-K) 
    elif transform.lower() in ('be'): 
          sigma = SSR/(N-K) 
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma, cov, se


def print_table(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values"],
        title="Results",
        **kwargs
    ) -> None:
    label_y, label_x = labels
    # Create table for data on coefficients
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i]
        ]
        table.append(row)
    
    # Print table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print data for model specification
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma').item():.3f}")
    
    
def perm( Q_T: np.ndarray, A: np.ndarray, t=0) -> np.ndarray:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.array): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.array): The vector or matrix that is to be transformed. Has
        to be a 2d array.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer t from the shape of the transformation matrix.
    if t==0:
        t = Q_T.shape[1]

    # Initialize the numpy array
    Z = np.array([[]])
    Z = Z.reshape(0, A.shape[1])

    # Loop over the individuals, and permutate their values.
    for i in range(int(A.shape[0]/t)):
        Z = np.vstack((Z, Q_T@A[i*t: (i + 1)*t]))
    return Z
