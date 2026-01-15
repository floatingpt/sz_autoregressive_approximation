"""
Time-Varying Multivariate Autoregressive (MVAR) Classifier

Binary classification for locally stationary multivariate time series based on 
second-order structure using MVAR approximations with sieve basis expansion.

Mathematical Framework:
----------------------
Each observation Z_t ∈ R^p follows a time-varying MVAR(b) model:

    Z_t = Σ_{j=1}^b A_j(t) Z_{t-j} + ε_t

where A_j(t) ∈ R^{p×p} are smooth coefficient matrices approximated using 
a sieve basis (B-splines or polynomial basis).

Feature Construction:
--------------------
For each lag j, compute discriminative features based on matrix norms:

    D(j) = sup_{t1,t2 ∈ [0,1]} ||Â_j(t1) - Â_j(t2)||

Aggregate features across upper lags:

    S = sup_{j ∈ upper-lag range} D(j)

Classification:
--------------
1. Compute per-series features S_k
2. Compute class medians S̄_1, S̄_2
3. Find threshold τ* that maximizes training accuracy
4. Classify new series based on distance to class medians

Robustness Features:
-------------------
- Handles unequal time series lengths via normalized time indices
- Robust to small/imbalanced samples via median-based features
- Stable under weak nonstationarity via regularization
"""

from __future__ import annotations

from typing import Literal, Tuple
import warnings

import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class TimeVaryingMVAR:
    """
    Time-varying Multivariate Autoregressive model with sieve basis expansion.
    
    Parameters
    ----------
    order : int
        AR model order (lag b)
    n_basis : int
        Number of basis functions for time-varying coefficients
    basis_type : {'bspline', 'polynomial'}
        Type of sieve basis
    regularization : float
        Ridge regularization parameter (0 = OLS)
    degree : int
        Degree for basis functions (B-spline degree or polynomial degree)
    """
    
    def __init__(
        self,
        order: int = 3,
        n_basis: int = 10,
        basis_type: Literal['bspline', 'polynomial'] = 'bspline',
        regularization: float = 0.1,
        degree: int = 3,
    ):
        self.order = order
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.regularization = regularization
        self.degree = degree
        
        self.n_channels = None
        self.coefficients_ = None  # Shape: (order, p, p, n_basis)
        self.scaler_ = StandardScaler()
        
    def _construct_basis(self, t: np.ndarray) -> np.ndarray:
        """
        Construct basis matrix Φ(t) ∈ R^{T × K} where K = n_basis.
        
        Parameters
        ----------
        t : array-like, shape (T,)
            Normalized time indices in [0, 1]
            
        Returns
        -------
        basis : ndarray, shape (T, n_basis)
            Basis function evaluations
        """
        t = np.asarray(t).ravel()
        n_time = len(t)
        
        if self.basis_type == 'bspline':
            # Construct B-spline basis with uniform knots
            knots = np.linspace(0, 1, self.n_basis - self.degree + 1)
            # Augment knots for full B-spline basis
            full_knots = np.concatenate([
                [0] * self.degree,
                knots,
                [1] * self.degree
            ])
            
            basis = np.zeros((n_time, self.n_basis))
            for i in range(self.n_basis):
                # Construct B-spline coefficients (indicator for basis i)
                c = np.zeros(self.n_basis)
                c[i] = 1.0
                spline = BSpline(full_knots, c, self.degree, extrapolate=False)
                basis[:, i] = spline(t)
                
        elif self.basis_type == 'polynomial':
            # Legendre-style orthogonal polynomial basis
            basis = np.zeros((n_time, self.n_basis))
            for k in range(self.n_basis):
                basis[:, k] = np.power(2 * t - 1, k)
                
        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}")
            
        return basis
    
    def _construct_design_matrix(
        self, 
        Z: np.ndarray, 
        basis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct design matrix for MVAR estimation.
        
        Parameters
        ----------
        Z : ndarray, shape (T, p)
            Multivariate time series
        basis : ndarray, shape (T, n_basis)
            Basis evaluations at each time point
            
        Returns
        -------
        X_design : ndarray, shape (T - order, p * order * n_basis)
            Design matrix with lagged values weighted by basis functions
        y : ndarray, shape (T - order, p)
            Response variables
        """
        T, p = Z.shape
        T_eff = T - self.order
        n_features = p * self.order * self.n_basis
        
        X_design = np.zeros((T_eff, n_features))
        y = Z[self.order:]
        
        # For each time t in [order, T)
        for t_idx in range(T_eff):
            t = t_idx + self.order
            
            # Stack lagged values and basis functions
            feature_idx = 0
            for lag_j in range(1, self.order + 1):
                # Z_{t-j} ∈ R^p
                z_lag = Z[t - lag_j]
                
                # Basis functions at time t: Φ(t) ∈ R^K
                phi_t = basis[t]
                
                # Outer product: z_lag ⊗ phi_t ∈ R^{p * K}
                for ch in range(p):
                    for k in range(self.n_basis):
                        X_design[t_idx, feature_idx] = z_lag[ch] * phi_t[k]
                        feature_idx += 1
                        
        return X_design, y
    
    def fit(self, Z: np.ndarray) -> TimeVaryingMVAR:
        """
        Fit time-varying MVAR model via regularized least squares.
        
        Parameters
        ----------
        Z : ndarray, shape (T, p)
            Multivariate time series (channels × time transposed to time × channels)
            
        Returns
        -------
        self : TimeVaryingMVAR
            Fitted model
        """
        if Z.ndim != 2:
            raise ValueError("Z must be 2D array (time × channels)")
            
        T, p = Z.shape
        self.n_channels = p
        
        # Normalize time indices
        t_norm = np.linspace(0, 1, T)
        
        # Construct basis
        basis = self._construct_basis(t_norm)
        
        # Standardize data (important for regularization)
        Z_scaled = self.scaler_.fit_transform(Z)
        
        # Construct design matrix
        X_design, y = self._construct_design_matrix(Z_scaled, basis)
        
        # Fit coefficients for each output channel separately
        self.coefficients_ = np.zeros((self.order, p, p, self.n_basis))
        
        for out_ch in range(p):
            if self.regularization > 0:
                ridge = Ridge(alpha=self.regularization, fit_intercept=False)
                ridge.fit(X_design, y[:, out_ch])
                theta = ridge.coef_
            else:
                # OLS solution -> consider penalty term
                theta = linalg.lstsq(X_design, y[:, out_ch])[0]
            
            # Reshape theta back to (order, p, n_basis)
            theta_reshaped = theta.reshape(self.order, p, self.n_basis)
            self.coefficients_[:, out_ch, :, :] = theta_reshaped
            
        return self
    
    def get_coefficient_matrices(self, t_eval: np.ndarray | None = None) -> np.ndarray:
        """
        Evaluate coefficient matrices A_j(t) at specified time points.
        
        Parameters
        ----------
        t_eval : array-like, shape (M,), optional
            Normalized time points in [0, 1] for evaluation.
            If None, uses uniform grid of 100 points.
            
        Returns
        -------
        A_matrices : ndarray, shape (M, order, p, p)
            Coefficient matrices A_j(t) for j=1,...,order and times in t_eval
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted first")
            
        if t_eval is None:
            t_eval = np.linspace(0, 1, 100)
        else:
            t_eval = np.asarray(t_eval).ravel()
            
        M = len(t_eval)
        basis = self._construct_basis(t_eval)  # Shape: (M, n_basis)
        
        A_matrices = np.zeros((M, self.order, self.n_channels, self.n_channels))
        
        # A_j(t) = Σ_k c_{j,k} φ_k(t)
        for m in range(M):
            for j in range(self.order):
                # Matrix multiply: coefficients[:, :, :, k] @ basis[m, k]
                A_matrices[m, j] = np.einsum(
                    'ijk,k->ij',
                    self.coefficients_[j],  # (p, p, n_basis)
                    basis[m]  # (n_basis,)
                )
                
        return A_matrices


class MVARFeatureExtractor:
    """
    Extract discriminative features from time-varying MVAR coefficients.
    
    Features are based on temporal variation in coefficient matrices:
        D(j) = sup_{t1,t2} ||A_j(t1) - A_j(t2)||
        
    Aggregated across upper lags:
        S = sup_{j ∈ upper_lags} D(j)
    """
    
    def __init__(
        self,
        mvar_order: int = 3,
        n_basis: int = 10,
        basis_type: Literal['bspline', 'polynomial'] = 'bspline',
        regularization: float = 0.1,
        upper_lag_range: Tuple[int, int] | None = None,
        norm_type: Literal['fro', 'spectral', 'operator'] = 'fro',
        n_time_points: int = 50,
    ):
        """
        Parameters
        ----------
        mvar_order : int
            MVAR model order
        n_basis : int
            Number of basis functions
        basis_type : {'bspline', 'polynomial'}
            Sieve basis type
        regularization : float
            Ridge regularization strength
        upper_lag_range : tuple of int, optional
            (start, end) indices for upper lag aggregation.
            If None, uses last half of lags.
        norm_type : {'fro', 'spectral', 'operator'}
            Matrix norm for computing D(j)
        n_time_points : int
            Number of time points for evaluating supremum
        """
        self.mvar_order = mvar_order
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.regularization = regularization
        self.upper_lag_range = upper_lag_range
        self.norm_type = norm_type
        self.n_time_points = n_time_points
        
    def _matrix_norm(self, A: np.ndarray) -> float:
        """Compute specified matrix norm."""
        if self.norm_type == 'fro':
            return np.linalg.norm(A, ord='fro')
        elif self.norm_type == 'spectral':
            return np.linalg.norm(A, ord=2)
        elif self.norm_type == 'operator':
            # Operator norm = largest singular value
            return np.linalg.norm(A, ord=2)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")
    
    def _compute_lag_variation(self, A_matrices: np.ndarray, lag_idx: int) -> float:
        """
        Compute D(j) = sup_{t1,t2} ||A_j(t1) - A_j(t2)||.
        
        Parameters
        ----------
        A_matrices : ndarray, shape (M, order, p, p)
            Coefficient matrices at M time points
        lag_idx : int
            Lag index j
            
        Returns
        -------
        d_j : float
            Maximum variation across time for lag j
        """
        M = A_matrices.shape[0]
        A_j = A_matrices[:, lag_idx]  # Shape: (M, p, p)
        
        # Compute all pairwise differences and their norms
        max_diff = 0.0
        for i in range(M):
            for j in range(i + 1, M):
                diff_norm = self._matrix_norm(A_j[i] - A_j[j])
                max_diff = max(max_diff, diff_norm)
                
        return max_diff
    
    def extract_features(self, Z: np.ndarray) -> float:
        """
        Extract scalar feature S from multivariate time series.
        
        Parameters
        ----------
        Z : ndarray, shape (T, p) or (p, T)
            Multivariate time series
            
        Returns
        -------
        S : float
            Aggregated discriminative feature
        """
        # Ensure Z is (T, p)
        if Z.shape[0] < Z.shape[1]:
            Z = Z.T
            
        # Fit MVAR model
        mvar = TimeVaryingMVAR(
            order=self.mvar_order,
            n_basis=self.n_basis,
            basis_type=self.basis_type,
            regularization=self.regularization,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mvar.fit(Z)
        
        # Get coefficient matrices at evaluation points
        t_eval = np.linspace(0, 1, self.n_time_points)
        A_matrices = mvar.get_coefficient_matrices(t_eval)
        
        # Determine upper lag range
        if self.upper_lag_range is None:
            # Use last half of lags
            start = self.mvar_order // 2
            end = self.mvar_order
        else:
            start, end = self.upper_lag_range
            
        # Compute D(j) for each lag in upper range
        lag_variations = []
        for j in range(start, end):
            d_j = self._compute_lag_variation(A_matrices, j)
            lag_variations.append(d_j)
            
        # Aggregate: S = sup_j D(j)
        S = np.max(lag_variations) if lag_variations else 0.0
        
        return S


class MVARBinaryClassifier:
    """
    Binary classifier for multivariate time series based on MVAR features.
    
    Implements Algorithm 1: Binary structural classification procedure
    
    Classification Algorithm (from Algorithm 1):
    ---------------------------------------------
    Step 1: For training time series {x_k1} (class 1) and {y_k2} (class 2),
            compute aggregated discriminative features S^x_k1, S^y_k2 based on
            temporal variation in MVAR coefficient matrices:
            
            For each series k and lag j:
              D_k(j) = sup_{t1,t2 ∈ [0,1]} ||Â_j^k(t1) - Â_j^k(t2)||
            
            Aggregate over upper lags:
              S_k = sup_{j ∈ upper-lag range} D_k(j)
            
            Compute average features:
              S̄_x = median({S^x_k1})
              S̄_y = median({S^y_k2})
    
    Step 2: Compute threshold value ϑ that maximizes training accuracy.
    
    Step 3: For test series z, compute aggregated discriminative feature S^z.
    
    Step 4: Classification rule based on threshold and class averages:
            - If S^z < ϑ:
                * Assign to class 1 when S̄_x < S̄_y
                * Assign to class 2 when S̄_x > S̄_y
            - If S^z > ϑ:
                * Assign to class 2 when S̄_x < S̄_y  
                * Assign to class 1 when S̄_x > S̄_y
    
    Multiple Channel Handling:
    --------------------------
    The multivariate AR model captures dependencies across all p channels:
    - Z_t ∈ R^p represents all channels at time t
    - Coefficient matrices A_j(t) ∈ R^{p×p} encode cross-channel dynamics
    - Matrix norms measure joint variation across all channel interactions
    - Features aggregate information from the full p×p coefficient structure
    """
    
    def __init__(
        self,
        mvar_order: int = 3,
        n_basis: int = 10,
        basis_type: Literal['bspline', 'polynomial'] = 'bspline',
        regularization: float = 0.1,
        upper_lag_range: Tuple[int, int] | None = None,
        norm_type: Literal['fro', 'spectral', 'operator'] = 'fro',
        n_time_points: int = 50,
        n_grid_points: int = 100,
        threshold_metric: Literal['f1_seizure', 'youden'] = 'f1_seizure',
        seizure_weight: float | None = None,
    ):
        """
        Parameters
        ----------
        mvar_order : int
            MVAR model order
        n_basis : int
            Number of basis functions
        basis_type : {'bspline', 'polynomial'}
            Sieve basis type
        regularization : float
            Ridge regularization for stability
        upper_lag_range : tuple, optional
            Lag range for feature aggregation
        norm_type : {'fro', 'spectral', 'operator'}
            Matrix norm type
        n_time_points : int
            Time resolution for supremum computation
        n_grid_points : int
            Grid resolution for threshold search
        """
        self.feature_extractor = MVARFeatureExtractor(
            mvar_order=mvar_order,
            n_basis=n_basis,
            basis_type=basis_type,
            regularization=regularization,
            upper_lag_range=upper_lag_range,
            norm_type=norm_type,
            n_time_points=n_time_points,
        )
        self.n_grid_points = n_grid_points
        self.threshold_metric = threshold_metric
        self.seizure_weight = seizure_weight
        
        # Learned parameters (Algorithm 1 notation)
        self.class_averages_ = None  # S̄_x (class 0) and S̄_y (class 1)
        self.threshold_ = None  # ϑ
        self.train_features_ = None  # {S^x_k1, S^y_k2}
        self.train_labels_ = None
        self.label_order_ = None  # Track which class has lower average
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> MVARBinaryClassifier:
        """
        Fit classifier on training data following Algorithm 1.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, p, T) or (n_samples, T, p)
            Training time series. Multiple channels (p) capture EEG/multivariate data.
        y : ndarray, shape (n_samples,)
            Binary labels {0, 1}
            
        Returns
        -------
        self : MVARBinaryClassifier
        """
        n_samples = X.shape[0]
        
        # Step 1: Extract aggregated discriminative features S_k for each series
        print("Step 1: Extracting MVAR features from training data...")
        print(f"  Series shape: {X[0].shape} (channels × time or time × channels)")
        features = np.zeros(n_samples)
        for i in range(n_samples):
            # Feature extraction captures cross-channel dynamics via p×p matrices
            features[i] = self.feature_extractor.extract_features(X[i])
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_samples} series")
                
        self.train_features_ = features
        self.train_labels_ = y
        
        # Compute class average features S̄_x and S̄_y (using median for robustness)
        features_class0 = features[y == 0]
        features_class1 = features[y == 1]
        
        self.class_averages_ = {
            0: np.median(features_class0),  # S̄_x (or S̄_y depending on labeling)
            1: np.median(features_class1),
        }
        
        # Track ordering: which class has lower average feature value
        self.label_order_ = 'ascending' if self.class_averages_[0] < self.class_averages_[1] else 'descending'
        
        print(f"\nClass average features (S̄):")
        print(f"  Class 0: {self.class_averages_[0]:.4f}")
        print(f"  Class 1: {self.class_averages_[1]:.4f}")
        print(f"  Order: {self.label_order_}")
        
        # Step 2: Find optimal threshold ϑ via grid search (maximizes training accuracy)
        print("\nStep 2: Finding optimal threshold ϑ...")
        feature_min = features.min()
        feature_max = features.max()
        thresholds = np.linspace(feature_min, feature_max, self.n_grid_points)
        
        best_score = -np.inf
        best_thresh = thresholds[0]

        def _score(pred: np.ndarray, y_true: np.ndarray) -> float:
            tp = np.sum((pred == 1) & (y_true == 1))
            fp = np.sum((pred == 1) & (y_true == 0))
            fn = np.sum((pred == 0) & (y_true == 1))
            tn = np.sum((pred == 0) & (y_true == 0))

            # Avoid div-by-zero
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1_seiz = 2 * precision * recall / (precision + recall + 1e-9)

            sens = recall  # seizure recall
            spec = tn / (tn + fp + 1e-9)
            youden = sens + spec - 1

            if self.threshold_metric == 'youden':
                if self.seizure_weight is not None:
                    # weighted version: w*sens + (1-w)*spec
                    w = self.seizure_weight
                    return w * sens + (1 - w) * spec
                return youden
            # default: f1_seizure
            if self.seizure_weight is not None:
                # small bias toward seizure recall
                return f1_seiz + self.seizure_weight * sens * 0.01
            return f1_seiz

        for thresh in thresholds:
            pred = self._classify_by_threshold(features, thresh)
            score = _score(pred, y)
            if score > best_score:
                best_score = score
                best_thresh = thresh
                
        self.threshold_ = best_thresh
        
        print(f"  Optimal threshold ϑ: {self.threshold_:.4f}")
        print(f"  Training {self.threshold_metric} score: {best_score:.4f}")
        
        return self
    
    def _classify_by_threshold(self, features: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply Algorithm 1 Step 4 classification rule.
        
        Classification logic:
        - If S < ϑ: assign to class with lower average
        - If S > ϑ: assign to class with higher average
        
        Parameters
        ----------
        features : ndarray
            Feature values S for samples
        threshold : float
            Threshold value ϑ
            
        Returns
        -------
        predictions : ndarray
            Predicted class labels {0, 1}
        """
        predictions = np.zeros(len(features), dtype=int)
        
        # Determine which class to assign based on feature value relative to threshold
        if self.label_order_ == 'ascending':
            # Class 0 has lower average, class 1 has higher average
            # S < ϑ → class 0, S > ϑ → class 1
            predictions = (features > threshold).astype(int)
        else:
            # Class 0 has higher average, class 1 has lower average
            # S < ϑ → class 1, S > ϑ → class 0
            predictions = (features < threshold).astype(int)
            
        return predictions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for new time series (Algorithm 1 Steps 3-4).
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, p, T) or (n_samples, T, p)
            Test time series with p channels
            
        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted binary labels
        """
        if self.class_averages_ is None:
            raise ValueError("Classifier must be fitted first")
            
        n_samples = X.shape[0]
        features = np.zeros(n_samples)
        
        # Step 3: Compute aggregated discriminative feature S^z for each test series
        print(f"Step 3: Extracting features from {n_samples} test series...")
        for i in range(n_samples):
            features[i] = self.feature_extractor.extract_features(X[i])
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_samples} series")
        
        # Step 4: Apply threshold-based classification rule
        print(f"Step 4: Classifying based on threshold ϑ = {self.threshold_:.4f}...")
        y_pred = self._classify_by_threshold(features, self.threshold_)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (soft assignment based on distance to threshold).
        
        Uses distance from feature value to threshold, normalized by distance
        to class averages, to create probabilistic predictions.
        
        Parameters
        ----------
        X : ndarray
            Test time series
            
        Returns
        -------
        proba : ndarray, shape (n_samples, 2)
            Class probabilities
        """
        if self.class_averages_ is None:
            raise ValueError("Classifier must be fitted first")
            
        n_samples = X.shape[0]
        features = np.zeros(n_samples)
        
        for i in range(n_samples):
            features[i] = self.feature_extractor.extract_features(X[i])
        
        # Soft assignment based on distance to class averages
        dist_0 = np.abs(features - self.class_averages_[0])
        dist_1 = np.abs(features - self.class_averages_[1])
        total_dist = dist_0 + dist_1 + 1e-10
        proba = np.zeros((n_samples, 2))

        if self.label_order_ == 'ascending':
            # class 0 has lower average
            proba[:, 0] = dist_1 / total_dist
            proba[:, 1] = dist_0 / total_dist
        else:
            # class 1 has lower average
            proba[:, 1] = dist_0 / total_dist
            proba[:, 0] = dist_1 / total_dist
        
        return proba
