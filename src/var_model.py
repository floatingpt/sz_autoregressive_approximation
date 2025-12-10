from mat73 import loadmat # use for io with .73 mat files
import matplotlib
from sklearn.decomposition import PCA
import numpy as np
from statsmodels.tsa import api
import joblib
import scipy as sc


def fit_var_window(X_window, p): # (electrode channel x sample)
    
   
    data = X_window.T  # 
    model = api.VAR(data)
    # small windows create singularization
    try:
        res = model.fit(maxlags=p, ic=None)
        used_p = res.k_ar # p lags 
        A = []
        # res.params (k_ar*neqs + const) shape
        coefs = res.coefs  # (used_p, neqs, neqs)
        for k in range(used_p):
            A.append(coefs[k])  # (neqs, neqs)
        Sigma = res.sigma_u  # residual cov (neqs, neqs)
    except Exception as e:
        # least squares 
        # build lag matrix
        T, n = data.shape[0], data.shape[1]
        Y = data[p:]
        X = np.hstack([data[p - k - 1:-k - 1] for k in range(p)])
        # solve coef approximation
        B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        # reshape B to (p, n, n)
        coefs = B.T.reshape(n, p, n).transpose(1,0,2)
        A = [coefs[k] for k in range(p)]
        resid = Y - X.dot(B)
        Sigma = np.cov(resid.T)
        used_p = p
    return A, Sigma, used_p

def var_features_from_A_Var(A_list, Sigma):
    # flatten coefficients
    if len(A_list) > 0:
        coeff_vec = np.concatenate([A.flatten() for A in A_list])
    else:
        coeff_vec = np.array([])
    # logm of sigma (symmetric)
    Sigma_log = sc.linalg.logm(Sigma)
    # triu vectorization
    iu = np.triu_indices(Sigma_log.shape[0])
    sigma_vec = Sigma_log[iu]
    feat = np.concatenate([coeff_vec.real, sigma_vec.real])
    return feat