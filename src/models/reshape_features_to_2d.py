import numpy as np


def reshape_features_to_2D(X):
    """
    This function is a wrapper around np.reshape. It is intended to be used as a FunctionTransformer in sklearn/skorch pipelines.

    It reshapes a 3D array to 2D by concatenating the second and third dimensions of the input array. The order of the elements in the reshaped array is preserved using Fortran-like index ordering.

    Parameters
    ----------
    X : ndarray
        The input array to be reshaped. It should be a 3D array.

    Returns
    -------
    ndarray
        The reshaped 2D array. The shape of the returned array is (X.shape[0], X.shape[1]*X.shape[2]).

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
