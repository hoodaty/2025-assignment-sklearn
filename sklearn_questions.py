"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:
- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.

Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:
from sklearn.metrics.pairwise import pairwise_distances
to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances
from pandas.api.types import is_datetime64_any_dtype


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier.

    A very small kNN classifier using Euclidean distance. The estimator
    validates inputs using scikit-learn helpers so it is compatible with
    `check_estimator`.
    """

    def __init__(self, n_neighbors=1):
        """Initialize KNearestNeighbors with number of neighbors.

        Parameters
        ----------
        n_neighbors : int, default=1
            Number of nearest neighbors to use.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        X, y = validate_data(self, X, y, ensure_2d=True, allow_nd=False)
        check_classification_targets(y)

        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        self.classes_, y_encoded = np.unique(self.y_, return_inverse=True)
        self.y_encoded_ = y_encoded

        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self, ['X_', 'y_', 'classes_', 'y_encoded_'])
        X = validate_data(self, X, ensure_2d=True, allow_nd=False, reset=False)

        dists = pairwise_distances(X, self.X_)
        k = self.n_neighbors

        # indices of k nearest neighbors
        neigh_idx = np.argsort(dists, axis=1)[:, :k]

        # get encoded neighbor labels
        neigh_labels = np.take(self.y_encoded_, neigh_idx)

        def majority_vote(row):
            counts = np.bincount(row, minlength=self.classes_.shape[0])
            return counts.argmax()

        votes = np.apply_along_axis(majority_vote, 1, neigh_labels)
        y_pred = self.classes_[votes]
        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        -------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        check_is_fitted(self, ["X_", "y_", "classes_", "y_encoded_"])
        y_pred = self.predict(X)
        y = np.asarray(y)
        return float(np.mean(y_pred == y))


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, default='index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):
        """Initialize MonthlySplit with time column.

        Parameters
        ----------
        time_col : str, default='index'
            Column name or 'index' to use for datetime splitting.
        """
        self.time_col = time_col

    def __repr__(self):
        """Return string representation of MonthlySplit."""
        return f"MonthlySplit(time_col='{self.time_col}')"

    def _get_times(self, X):
        """Extract datetime series from DataFrame or index.

        Parameters
        ----------
        X : DataFrame
            Input data with datetime index or column.

        Returns
        -------
        pd.Series
            Series containing datetime values.

        Raises
        ------
        ValueError
            If X has no index or column, or if datetime column is invalid.
        """
        if self.time_col == 'index':
            if not hasattr(X, 'index'):
                raise ValueError('X has no index to use as datetime')
            times = X.index
        else:
            if not isinstance(X, pd.DataFrame):
                msg = 'When using a column name, X must be a DataFrame'
                raise ValueError(msg)
            if self.time_col not in X.columns:
                raise ValueError(f"Column '{self.time_col}' not found in X")
            times = X[self.time_col]

        if not is_datetime64_any_dtype(times):
            msg = "time column or index must be datetime64 type"
            raise ValueError(msg)
        return pd.Series(times)

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : DataFrame
            Input data with datetime index or column.
        y : array-like, optional
            Target values (unused).
        groups : array-like, optional
            Group labels (unused).

        Returns
        -------
        int
            Number of splits (n_months - 1).
        """
        times = self._get_times(X)
        periods = times.dt.to_period('M')
        unique_periods = np.sort(np.unique(periods))
        return max(0, len(unique_periods) - 1)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : DataFrame
            Input data with datetime index or column.
        y : array-like, optional
            Target values (unused).
        groups : array-like, optional
            Group labels (unused).

        Yields
        ------
        tuple
            Tuple of (train_indices, test_indices) for each successive month.
        """
        times = self._get_times(X)
        periods = times.dt.to_period('M')
        unique_periods = np.sort(np.unique(periods))

        for i in range(len(unique_periods) - 1):
            train_period = unique_periods[i]
            test_period = unique_periods[i + 1]
            idx_train = np.where(periods == train_period)[0].astype(int)
            idx_test = np.where(periods == test_period)[0].astype(int)
            yield (idx_train, idx_test)