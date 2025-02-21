import numpy as np
from numpy import linalg as la
from tabulate import tabulate

def estimate(y, x, z=None, transform='', T=None, robust_se=False):
    # Check input dimensions
    assert y.ndim == 2, 'Input y is 2D'
    assert x.ndim == 2, 'Input x is 2D'
    assert y.shape[1] == 1, 'y must be column vector'
    assert y.shape[0] == x.shape[0], 'y and x have the same first dimension'
    
    DOIV = z is not None
    
    # Choose estimation method
    if DOIV:
        b_hat = est_piv(y, x, z)
    else:
        b_hat = est_ols(y, x)
    
    residual = y - np.dot(x, b_hat)
    SSR = np.sum(residual ** 2)
    SST = np.sum((y - y.mean()) ** 2)
    R2 = 1.0 - (SSR / SST)
    
    if DOIV:
        gammahat = la.solve(np.dot(z.T, z), np.dot(z.T, x))
        x_ = np.dot(z, gammahat)
        R2 = np.nan  # Reset R2 when using IV 
    else:
        x_ = x
    
    sigma2, cov, se = variance(transform, SSR, x_, T)
    if robust_se:
        cov, se = robust(x_, residual, T)
    
    t_values = b_hat / se
    
    results = {
        'b_hat': b_hat,
        'se': se,
        'sigma2': sigma2,
        't_values': t_values,
        'R2': R2,
        'cov': cov
    }
    
    return results

def est_ols(y, x):
    """ Estimates y on x using ordinary least squares (OLS). """
    XTX_inv = la.inv(np.dot(x.T, x))
    XTy = np.dot(x.T, y)
    return np.dot(XTX_inv, XTy)

def est_piv(y, x, z):
    """ Estimates y on x using instrumental variables (2SLS). """
    gamma = la.inv(np.dot(z.T, z)) @ (z.T @ x)
    xh = np.dot(z, gamma)
    betahat = la.inv(np.dot(xh.T, xh)) @ (xh.T @ y)
    return betahat

def variance(transform, SSR, x, T):
    """ Calculates variance and standard errors for OLS estimates. """
    NT, K = x.shape
    
    if transform in ('', 'fd', 'be', 're'):
        df = K
    elif transform == 'fe':
        N = NT // T
        df = N + K
    else:
        raise ValueError(f'Transform "{transform}" not implemented.')
    
    sigma2 = SSR / (NT - df)
    cov = sigma2 * la.inv(np.dot(x.T, x))
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    
    return sigma2, cov, se

def robust(x, residual, T):
    """ Computes robust standard errors. """
    NT, K = x.shape
    
    if T is None or T == 1:
        Ainv = la.inv(np.dot(x.T, x))
        uhat2_x = residual**2 * x  # Element-wise multiplication
        cov = Ainv @ np.dot(x.T, uhat2_x) @ Ainv
    else:
        N = NT // T
        B = np.zeros((K, K))
        for i in range(N):
            idx = slice(i*T, (i+1)*T)
            Omega = np.dot(residual[idx], residual[idx].T)
            B += np.dot(x[idx].T, np.dot(Omega, x[idx]))
        Ainv = la.inv(np.dot(x.T, x))
        cov = Ainv @ B @ Ainv
    
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return cov, se

def print_table(labels, results, headers=None, title="Results", _lambda=None, **kwargs):
    """ Prints regression results in tabular format. """
    if headers is None:
        headers = ["", "Beta", "Se", "t-values"]
    
    label_y, label_x = labels
    assert len(label_x) == results['b_hat'].size, "Mismatch between labels and estimated parameters."
    
    table = []
    for i, name in enumerate(label_x):
        row = [name, results['b_hat'][i], results['se'][i], results['t_values'][i]]
        table.append(row)
    
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    print(f"R² = {results['R2']:.3f}")
    print(f"σ² = {results['sigma2']:.3f}")
    if _lambda:
        print(f"λ = {_lambda:.3f}")

def perm(Q_T, A):
    """ Applies transformation matrix Q_T to A. """
    M, T = Q_T.shape
    NT, K = A.shape
    N = NT // T
    Z = np.zeros((M * N, K))
    
    for i in range(N):
        Z[i*M:(i+1)*M, :] = np.dot(Q_T, A[i*T:(i+1)*T, :])
    
    return Z

def load_example_data():
    """ Loads example dataset from 'wagepan.txt'. """
    data = np.loadtxt('wagepan.txt', delimiter=",")
    id_array = data[:, 0]
    unique_id = np.unique(id_array, return_counts=True)
    n = unique_id[0].size
    t = int(unique_id[1].mean())
    year = data[:, 1].astype(int)
    
    y = data[:, 8].reshape(-1, 1)
    x = np.column_stack([
        np.ones(y.shape[0]),
        data[:, 2],
        data[:, 4],
        data[:, 6],
        data[:, 3],
        data[:, 9],
        data[:, 5],
        data[:, 7]
    ])
    
    label_y = 'Log wage'
    label_x = [
        'Constant', 'Black', 'Hispanic', 'Education', 'Experience',
        'Experience sqr', 'Married', 'Union'
    ]
    
    return y, x, n, t, year, label_y, label_x
