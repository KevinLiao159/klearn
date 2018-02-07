"""
This module is responsible for trivial tasks, such as checking
data, making assertions, converting to numpy for calculations
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection._split import check_cv


__all__ = ('check_gravity_index',
           'force_array',
           'check_consistent_length',
           'check_is_fitted',
           'process_cv_results',
           'save_object',
           'check_cv')


def check_gravity_index(df):
    assert df.index.names == ['date', 'tradingitemid']


def force_array(array):
    """
    Parameters
    ----------
    array : object
        Input object to check / convert.

    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    if isinstance(array, pd.DataFrame) and array.shape[1] == 1:
        array = check_array(
            array.iloc[:, 0],
            force_all_finite=False,
            ensure_2d=False
        )
    return np.asarray(array)


def check_consistent_length(*arrays):
    """
    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """
    lengths = [force_array(X).shape[0] for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def process_cv_results(cv_results):
    """
    This function reformats the .cv_results_ attribute of a fitted randomized
    search (or grid search) into a dataframe with only the columns we care
    about.

    Args
    --------------
    cv_results : the .cv_results_ attribute of a fitted randomized search
    (or grid search) object

    Returns
    --------------
    a sorted dataframe with select information

    """
    results = pd.DataFrame(cv_results)
    cols = ['mean_test_score', 'mean_train_score', 'std_test_score']
    cols += [c for c in results.columns.values if c.startswith('param_')]
    return results[cols].sort_values(by='mean_test_score', ascending=False)


def save_object(obj, filepath):
    with open(filepath, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filepath):
    with open(filepath, 'rb') as input:
        obj = pickle.load(input)
    return obj
