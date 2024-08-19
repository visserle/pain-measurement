import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


def reshape_features_to_2D(X):
    """
    This function is a wrapper around np.reshape. It is intended to be used as a
    FunctionTransformer in sklearn/skorch pipelines.

    It reshapes a 3D array to 2D by concatenating the second and third dimensions of the
    input array. The order of the elements in the reshaped array is preserved using
    Fortran-like index ordering.

    Parameters
    ----------
    X : ndarray
        The input array to be reshaped. It should be a 3D array.

    Returns
    -------
    ndarray
        The reshaped 2D array. The shape of the returned array is
        (X.shape[0], X.shape[1]*X.shape[2]).

    Raises
    ------
    ValueError
        If the input array `X` is not a 3D array.

    Examples
    --------
    >>> reshape_features_to_2D(np.random.rand(10, 5, 5)).shape
    (10, 25)

    In a sklearn/skorch pipeline:
    ```python
    from sklearn.preprocessing import FunctionTransformer

    mlp_pipe = make_pipeline(
        StandardScaler3D(), FunctionTransformer(reshape_features_to_2D), mlp
    )
    ```
    """
    if len(X.shape) != 3:
        raise ValueError("Input array should be a 3D array.")
    return np.reshape(
        X,
        (
            X.shape[0],
            X.shape[1] * X.shape[2],  # same as -1
        ),
        order="F",
    )


"""A module used to represent different sklearn scalers for 3D data."""


class Scaler3D(BaseEstimator, TransformerMixin):
    """
    A class used to represent a 3D scaler.

    This class is a wrapper around various scikit-learn scalers, designed to work with
    3D data. The scaler instance can then be used on new data to transform it the same
    way it did on the training set.

    Attributes
    ----------
    copy : bool
        Determines whether the input data should be copied, or overwritten when
        transformed (default is True). If False, try to avoid a copy and do inplace
        scaling instead. This is not guaranteed to always work inplace; e.g. if the data
        is not a NumPy array or scipy.sparse CSR matrix, a copy may still be returned.
    scaler : object
        The scaler object to be used for scaling the data.
    is_fitted : bool
        Tracks if the scaler has been fitted.

    Methods
    -------
    fit(X, y=None)
        Fits the scaler to the data.
    transform(X)
        Transforms the data using the fitted scaler.
    fit_transform(X, y=None)
        Fits the scaler to the data and then transforms the data.
    inverse_transform(X)
        Transforms the data back to its original space using the fitted scaler.
    _validate_input(X)
        Validates the input data.
    """

    def __init__(self, scaler, copy=True):
        """
        Parameters
        ----------
        scaler : object
            The scaler object to be used for scaling the data.
        copy : bool, optional
            Determines whether the input data should be copied, or overwritten when
            transformed (default is True).
        """
        self.copy = copy  # necessary for sklearn pipelines to work
        self.scaler = scaler
        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Fits the scaler to the data.

        Parameters
        ----------
        X : array-like
            The data to be used for fitting the scaler.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_input(X)
        self.scaler.fit(X.reshape(-1, X.shape[-1]))
        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted scaler.

        Parameters
        ----------
        X : array-like
            The data to be transformed.

        Returns
        -------
        X_transformed : array-like
            The transformed data.

        Raises
        ------
        ValueError
            If the scaler is not yet fitted to the data.
        """
        self._validate_input(X)
        if not self.is_fitted:
            raise ValueError(
                "The scaler is not fitted yet. Call 'fit' with appropriate arguments "
                "before using this estimator."
            )
        X_transformed = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        return X_transformed.reshape(X.shape)

    def fit_transform(self, X, y=None):
        """
        Fits the scaler to the data and then transforms the data.

        Parameters
        ----------
        X : array-like
            The data to be fitted and transformed.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_transformed : array-like
            The fitted and transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """
        Transforms the data back to its original space using the fitted scaler.

        Parameters
        ----------
        X : array-like
            The data to be inverse transformed.

        Returns
        -------
        X_inversed : array-like
            The inverse transformed data.

        Raises
        ------
        ValueError
            If the scaler is not yet fitted to the data.
        """
        self._validate_input(X)
        if not self.is_fitted:
            raise ValueError(
                "The scaler is not fitted yet. Call 'fit' with appropriate arguments "
                "before using this estimator."
            )
        X_inversed = self.scaler.inverse_transform(X.reshape(-1, X.shape[-1]))
        return X_inversed.reshape(X.shape)

    def _validate_input(self, X):
        """
        Validates the input data.

        Parameters
        ----------
        X : array-like
            The data to be validated.

        Raises
        ------
        ValueError
            If the input data is not a 3D array.
        """
        if len(X.shape) != 3:
            raise ValueError("Input array should be a 3D array.")


class StandardScaler3D(Scaler3D):
    """
    A class used to represent a 3D standard scaler.

    This class is a wrapper around the StandardScaler from scikit-learn, designed to
    work with 3D data. The StandardScaler standardizes features by removing the mean and
     scaling to unit variance. The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples and `s` is the standard deviation of
    the training samples.

    Attributes
    ----------
    copy : bool
        Determines whether the input data should be copied, or overwritten when
        transformed (default is True).
    scaler : object
        The scaler object to be used for scaling the data.
    is_fitted : bool
        Tracks if the scaler has been fitted.

    Methods
    -------
    fit(X, y=None)
        Fits the scaler to the data.
    transform(X)
        Transforms the data using the fitted scaler.
    fit_transform(X, y=None)
        Fits the scaler to the data and then transforms the data.
    inverse_transform(X)
        Transforms the data back to its original space using the fitted scaler.
    _validate_input(X)
        Validates the input data.
    """

    def __init__(self, copy=True):
        super().__init__(StandardScaler(copy=copy), copy)


class RobustScaler3D(Scaler3D):
    """
    A class used to represent a 3D robust scaler.

    This class is a wrapper around the RobustScaler from scikit-learn, designed to work
    with 3D data. The RobustScaler scales features using statistics that are robust to
    outliers. This method removes the median and scales the data according to the
    quantile range (defaults to IQR: Interquartile Range). The IQR is the range between
    the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

    Attributes
    ----------
    copy : bool
        Determines whether the input data should be copied, or overwritten when
        transformed (default is True).
    scaler : object
        The scaler object to be used for scaling the data.
    is_fitted : bool
        Tracks if the scaler has been fitted.

    Methods
    -------
    fit(X, y=None)
        Fits the scaler to the data.
    transform(X)
        Transforms the data using the fitted scaler.
    fit_transform(X, y=None)
        Fits the scaler to the data and then transforms the data.
    inverse_transform(X)
        Transforms the data back to its original space using the fitted scaler.
    _validate_input(X)
        Validates the input data.
    """

    def __init__(self, copy=True):
        super().__init__(RobustScaler(copy=copy), copy)


class MinMaxScaler3D(Scaler3D):
    """
    A class used to represent a 3D Min-Max scaler.

    This class is a wrapper around the MinMaxScaler from scikit-learn, designed to work
    with 3D data. The MinMaxScaler transforms features by scaling each feature to a
    given range, typically between zero and one. The transformation is given by:

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))\n
        X_scaled = X_std * (max - min) + min

    where min, max are the feature range.

    Attributes
    ----------
    copy : bool
        Determines whether the input data should be copied, or overwritten when
        transformed (default is True).
    scaler : object
        The scaler object to be used for scaling the data.
    is_fitted : bool
        Tracks if the scaler has been fitted.

    Methods
    -------
    fit(X, y=None)
        Fits the scaler to the data.
    transform(X)
        Transforms the data using the fitted scaler.
    fit_transform(X, y=None)
        Fits the scaler to the data and then transforms the data.
    inverse_transform(X)
        Transforms the data back to its original space using the fitted scaler.
    _validate_input(X)
        Validates the input data.
    """

    def __init__(self, copy=True):
        super().__init__(MinMaxScaler(copy=copy), copy)


class MaxAbsScaler3D(Scaler3D):
    """
    A class used to represent a 3D Max Absolute scaler.

    This class is a wrapper around the MaxAbsScaler from scikit-learn, designed to work
    with 3D data. The MaxAbsScaler transforms features by scaling each feature to the
    range [-1, 1] by dividing each feature by its maximum absolute value. The
    transformation is given by:

        X_scaled = X / abs(X.max(axis=0))

    where X.max(axis=0) is the maximum absolute value of each feature. This scaler is
    especially useful for data that is already centered at zero or is sparse. Once
    fitted, the scaler instance can be used on new data to transform it the same way it
    did on the training set.

    Attributes
    ----------
    copy : bool
        Determines whether the input data should be copied, or overwritten when
        transformed (default is True).
    scaler : object
        The scaler object to be used for scaling the data.
    is_fitted : bool
        Tracks if the scaler has been fitted.

    Methods
    -------
    fit(X, y=None)
        Fits the scaler to the data.
    transform(X)
        Transforms the data using the fitted scaler.
    fit_transform(X, y=None)
        Fits the scaler to the data and then transforms the data.
    inverse_transform(X)
        Transforms the data back to its original space using the fitted scaler.
    _validate_input(X)
        Validates the input data.
    """

    def __init__(self, copy=True):
        super().__init__(MaxAbsScaler(copy=copy), copy)
