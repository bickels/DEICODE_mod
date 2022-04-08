import numpy as np
# from skbio.stats.composition import closure
# need to ignore log of zero warning
np.seterr(all='ignore')

def closure(mat, *, out=None):
    """
    Performs closure to ensure that all elements add up to 1.
    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components
    out : array_like or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None, a
        freshly-allocated array is returned.
    Returns
    -------
    array_like, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Raises
    ------
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.
    ValueError
       Raises an error if there is a row that has all zeros.
    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import closure
    >>> X = np.array([[2, 2, 6], [4, 4, 2]])
    >>> closure(X)
    array([[ 0.2,  0.2,  0.6],
           [ 0.4,  0.4,  0.2]])
    from: https://github.com/ntessore/composition_stats/blob/main/composition_stats/__init__.py
    """
    mat = np.atleast_2d(mat)
    if out is not None:
        out = np.atleast_2d(out)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    norm = mat.sum(axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Input matrix cannot have rows with all zeros")
    return np.divide(mat, norm, out=out).squeeze()

def rclr(mat):
    """

    The rclr procedure first log transform
    the nonzero values before centering the data
    we refer to this preprocessing procedure as
    the robust center log-ratio (rclr) (1) due to its
    ties to the clr (2) transform commonly used in
    compositional data analysis.

    Parameters
    ----------
    mat : array_like, float
       a matrix of counts where
       rows = components and
       columns = samples

    Returns
    -------
    numpy.ndarray
        rclr transformed matrix

    Raises
    ------
    ValueError
        Raises an error if values in array are negative
    ValueError
        Data-mat contains either np.inf or -np.inf
    ValueError
        Data-mat contains nans

    References
    ----------
    .. [1] Martino, Cameron, James T. Morton, Clarisse A. Marotz,
           Luke R. Thompson, Anupriya Tripathi, Rob Knight, and
           Karsten Zengler. 2019. “A Novel Sparse Compositional
           Technique Reveals Microbial Perturbations.”
           Edited by Josh D. Neufeld. mSystems 4 (1).
           https://doi.org/10.1128/mSystems.00016-19.

    .. [2] Pawlowsky-Glahn, Vera, Juan José Egozcue, and
           Raimon Tolosana-Delgado. 2015. Modeling and
           Analysis of Compositional Data. John Wiley & Sons.

    Examples
    --------
    >>> import numpy as np
    >>> from deicode.preprocessing import rclr
    >>> x = np.array([[1, 3, 4, 2, 0],
              [4, 0, 1, 2, 5]])
    >>> rclr(x)
    array([[-0.79, 0.3, 0.59, -0.1, nan],
           [0.46, nan, -0.92, -0.23, 0.69]])
    """

    # ensure array is at leadt 2D
    mat = np.atleast_2d(np.array(mat))
    # ensure no missing values
    if (mat < 0).any():
        raise ValueError('Array Contains Negative Values')
    # ensure no undefined values
    if np.count_nonzero(np.isinf(mat)) != 0:
        raise ValueError('Data-mat contains either np.inf or -np.inf')
    # ensure no missing values
    if np.count_nonzero(np.isnan(mat)) != 0:
        raise ValueError('Data-mat contains nans')
    # take the log of the sample centered data
    mat = np.log(closure(mat))
    # generate a mask of missing values
    mask = [True] * mat.shape[0] * mat.shape[1]
    mask = np.array(mat).reshape(mat.shape)
    mask[np.isfinite(mat)] = False
    # sum of rows (features)
    lmat = np.ma.array(mat, mask=mask)
    # perfrom geometric mean
    gm = lmat.mean(axis=-1, keepdims=True)
    # center with the geometric mean
    lmat = (lmat - gm).squeeze().data
    # mask the missing with nan
    lmat[~np.isfinite(mat)] = np.nan
    return lmat
