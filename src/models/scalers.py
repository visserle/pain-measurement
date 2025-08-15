from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class Scaler3D(BaseEstimator, TransformerMixin):
    """
    Base class for applying scikit-learn scalers to 3D data.

    Wraps various scikit-learn scalers to work with 3D data by reshaping
    appropriately during fit and transform operations.

    Attributes
    ----------
    copy : bool
        If True, creates copy of input data. If False, tries to perform inplace scaling.
    scaler : object
        The underlying scikit-learn scaler instance.
    is_fitted : bool
        Whether the scaler has been fitted.
    """

    def __init__(self, scaler, copy=True):
        """
        Parameters
        ----------
        scaler : object
            Scikit-learn scaler instance
        copy : bool, optional
            Whether to copy input data (default True)
        """
        self.copy = copy  # necessary for sklearn pipelines to work
        self.scaler = scaler
        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Fit scaler to 3D input data.

        Parameters
        ----------
        X : array-like
            Input data to fit
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self
        """
        self._validate_input(X)
        self.scaler.fit(X.reshape(-1, X.shape[-1]))
        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Transform 3D input data using fitted scaler.

        Parameters
        ----------
        X : array-like
            Data to transform

        Returns
        -------
        array-like
            Transformed data
        """
        self._validate_input(X)
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        X_transformed = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        return X_transformed.reshape(X.shape)

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """
        Transform data back to original space.

        Parameters
        ----------
        X : array-like
            Data to inverse transform

        Returns
        -------
        array-like
            Inverse transformed data
        """
        self._validate_input(X)
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        X_inversed = self.scaler.inverse_transform(X.reshape(-1, X.shape[-1]))
        return X_inversed.reshape(X.shape)

    def _validate_input(self, X):
        """Validate that input is 3D array"""
        if len(X.shape) != 3:
            raise ValueError("Input array should be a 3D array.")


class StandardScaler3D(Scaler3D):
    """
    3D wrapper for sklearn's StandardScaler.

    Standardizes features by removing mean and scaling to unit variance:
    z = (x - u) / s
    where u is mean and s is standard deviation of training samples.
    """

    def __init__(self, copy=True):
        super().__init__(StandardScaler(copy=copy), copy)


class RobustScaler3D(Scaler3D):
    """
    3D wrapper for sklearn's RobustScaler.

    Scales features using statistics robust to outliers by removing median
    and scaling using quantile range (default IQR).
    """

    def __init__(self, copy=True):
        super().__init__(RobustScaler(copy=copy), copy)


class MinMaxScaler3D(Scaler3D):
    """
    3D wrapper for sklearn's MinMaxScaler.

    Scales features to a fixed range (typically [0,1]) using:
    X_std = (X - X.min) / (X.max - X.min)
    X_scaled = X_std * (max - min) + min
    """

    def __init__(self, copy=True):
        super().__init__(MinMaxScaler(copy=copy), copy)


class MaxAbsScaler3D(Scaler3D):
    """
    3D wrapper for sklearn's MaxAbsScaler.

    Scales features to [-1,1] range by dividing through largest maximum
    absolute value of each feature: X_scaled = X / abs(X.max)
    Useful for data already centered at zero or sparse data.
    """

    def __init__(self, copy=True):
        super().__init__(MaxAbsScaler(copy=copy), copy)
