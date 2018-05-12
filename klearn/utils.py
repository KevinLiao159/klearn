"""
This module is responsible for trivial tasks, such as checking
data, making assertions, converting to numpy for calculations
"""

import os
import psutil
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from sklearn.utils import check_array, shuffle
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection._split import check_cv


__all__ = ('check_gravity_index',
           'force_array',
           'ensure_2d_array',
           'check_consistent_length',
           'fit_model',
           'check_has_set_attr',
           'check_is_fitted',
           'process_cv_results',
           'memory_stat',
           'save_object',
           'load_object',
           'check_cv',
           'shuffle')


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
    X_converted : numpy array
        The converted and validated X.
    """
    if isinstance(array, pd.DataFrame) and array.shape[1] == 1:
        array = check_array(
            array.iloc[:, 0],
            force_all_finite=False,
            ensure_2d=False
        )
    return np.asarray(array)


def ensure_2d_array(array, axis=1):
    """
    Parameters
    ----------
    array : object
        Input object to check / convert.

    axis : integer
        Position in the axes where the new axis is placed.

    Returns
    -------
    X_converted : 2d numpy array
        The converted and validated X.
    """
    X = force_array(array)
    if X.ndim == 1:
        if axis == 1:
            X = X.reshape(-1, 1)
        else:
            X = X.reshape(1, -1)
    return X


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


def fit_model(model, X, y, *args, **kwargs):
    model = deepcopy(model)
    model.fit(X, y, *args, **kwargs)
    return model


def check_has_set_attr(obj, attributes):
    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]
    for attr in attributes:
        if hasattr(obj, attr):
            pass
        else:
            raise AttributeError('Object does NOT have {}'.format(attr))


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
    if 'mean_train_score' not in cv_results.keys():
        cols = ['mean_test_score', 'std_test_score']
    cols += [c for c in results.columns.values if c.startswith('param_')]
    return results[cols].sort_values(by='mean_test_score', ascending=False)


def memory_stat():
    # memory
    process = psutil.Process(os.getpid())
    memused = process.memory_info().rss
    print('Total memory in use before reading data: {:.02f} GB '
          ''.format(memused/(2**30)))


def save_object(obj, filepath):
    with open(filepath, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filepath):
    with open(filepath, 'rb') as input:
        obj = pickle.load(input)
    return obj
