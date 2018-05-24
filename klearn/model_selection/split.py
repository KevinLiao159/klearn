"""
Module 'split' includes classes and
functions to split the data based on valid cross-validation strategy.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection._split import BaseCrossValidator
from gravity_learn.utils import force_array, check_consistent_length

__all__ = ['TSBFold',
           'ts_train_test_split',
           'group_train_test_split',
           'ts_predefined_split']

# --------------------------------------------------
#  Sci-kit Learn Like CV Classes
# --------------------------------------------------


class TSBFold(BaseCrossValidator):
    """
    This is a cross validation strategy that is mean to work with date vector
    provided. For the first of n_splits, the training window will be from
    the minimum date to the minimum date + size of the training window
    (train_t). The test set will start at minimum date + train_t + buff_t. The
    end of the test set will come test_t days after that. For the next split,
    we shift all these points up (by an amount determined by our total number
    and the number of splits we have.)

    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 1.

    train_window : int, default = 730 (days)
        The number of days in the training window

    buff_window : int, default = 380 (days)
        The number of days to skip between the end of the training window
        and the start of the testing window.

    test_window : int, default = 365 (days)
        The number of days in the testing window

    """
    def __init__(self, n_splits=3, train_window=730,
                 buff_window=380, test_window=365,
                 random_state=None):
        # TODO: Need to add random_state for sample dates
        self.n_splits = n_splits
        self.train_window = train_window
        self.buff_window = buff_window
        self.test_window = test_window

    def _return_winds(self, start):
        end_train = start + timedelta(days=self.train_window)
        start_test = end_train + timedelta(days=self.buff_window)
        end = start_test + timedelta(days=self.test_window)
        return end_train, start_test, end

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.

        groups (dates) : array-like, index, shape (n_samples, )
            with datetime type
            NOTE: dates have to be sorted (ascending)

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # check group
        if groups is None:
            raise ValueError('You have to pass in groups')
        if X is not None:
            check_consistent_length(force_array(X), force_array(groups))
        # get dates
        dates = groups
        tot_len = self.train_window + self.buff_window + self.test_window
        min_d = min(dates)
        max_d = max(dates)
        day_shift = ((max_d - min_d).days - tot_len)/(self.n_splits - 1)

        starts = \
            [min_d + timedelta(days=day_shift*i) for i in range(self.n_splits)]

        for start in starts:
            end_train, start_test, end = self._return_winds(start)
            # get id
            start_id = np.searchsorted(a=dates, v=start, side='left')
            end_train_id = np.searchsorted(a=dates, v=end_train, side='right')
            start_test_id = np.searchsorted(a=dates, v=start_test, side='left')
            end_id = np.searchsorted(a=dates, v=end, side='right')
            yield (
                np.arange(start=start_id, stop=end_train_id, step=1),          # noqa
                np.arange(start=start_test_id, stop=end_id, step=1))           # noqa

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


# --------------------------------------------------
#  Single Split CV Functions
# --------------------------------------------------

def ts_train_test_split(X=None, y=None, groups=None, train_window=None,
                        buff_window=380, test_window=365):
    """
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.

    y : array-like, shape (n_samples,)
        Always ignored, exists for compatibility.

    groups (dates) : array-like, index, shape (n_samples, )
        with datetime type

    train_window : int, default = 730 (days)
        The number of days in the training window

    buff_window : int, default = 380 (days)
        The number of days to skip between the end of the training window
        and the start of the testing window.

    test_window : int, default = 365 (days)
        The number of days in the testing window

    Returns
    -------
    train : ndarray
        The training set indices for that split.

    test : ndarray
        The testing set indices for that split.
    """
    # check group
    if groups is None:
        raise ValueError('You have to pass in groups')
    if X is not None:
        check_consistent_length(force_array(X), force_array(groups))
    # get min/max date
    dates = groups
    # min_d = min(dates)
    max_d = max(dates)
    # get test start date:
    test_start = max_d - timedelta(days=test_window)
    # get train end date
    train_end = test_start - timedelta(days=buff_window)
    # get train start date
    if train_window is None:
        train_start = min(dates)
    else:  # train_window is given
        train_start = train_end - timedelta(days=train_window)
    # get id
    train_start_id = np.searchsorted(a=dates, v=train_start, side='left')
    train_end_id = np.searchsorted(a=dates, v=train_end, side='right')
    test_start_id = np.searchsorted(a=dates, v=test_start, side='left')
    end_id = np.searchsorted(a=dates, v=max_d, side='right')
    # train, test list
    train = np.arange(start=train_start_id, stop=train_end_id, step=1)
    test = np.arange(start=test_start_id, stop=end_id, step=1)
    return [tuple((train, test))]


def group_train_test_split(X=None, y=None, groups=None,
                           train_size=None, random_state=None):
    """
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.

    y : array-like, shape (n_samples,)
        Always ignored, exists for compatibility.

    groups : array-like, index, shape (n_samples, )

    train_size : float, int, or None, default None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to 0.80.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    train : ndarray
        The training set indices for that split.

    test : ndarray
        The testing set indices for that split.
    """
    # check group
    if groups is None:
        raise ValueError('You have to pass in groups')
    if X is not None:
        check_consistent_length(force_array(X), force_array(groups))
    # get unique groups and n_groups
    groups = force_array(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    # convert train_size to int
    if train_size is None:
        train_size = 0.80
    if train_size < 1:
        train_size = int(train_size * n_groups)
    # sample groups_train
    if random_state:
        np.random.seed(random_state)
    groups_train = np.random.choice(
        a=unique_groups,
        size=train_size,
        replace=False
    )
    # train, test list
    train_filter = force_array(pd.DataFrame(groups).isin(groups_train))
    test_filter = np.logical_not(train_filter)
    train = np.where(train_filter)[0]
    test = np.where(test_filter)[0]
    return [tuple((train, test))]


# --------------------------------------------------
#  Rolling Window Operation CV Functions
# --------------------------------------------------

def ts_predefined_split(X=None, y=None, groups=None,
                        test_fold=None,
                        train_window=20 * 52,
                        buff_window=1 * 52,
                        test_window=1 * 52):
    """
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.

    y : array-like, shape (n_samples,)
        Always ignored, exists for compatibility.

    test_fold : array-like, a list of timestamps of the beginning of
        eaching rolling test fold. Default is None, it will take the
        latest available date, which is end_date - test_window

    groups (dates) : array-like, index, shape (n_samples, )
        with datetime type
        NOTE: for the time being, it will convert to 'datetime64[D]'
            we could support 'datetime64[ns]' in the future

    train_window : int, default = 20 * 52 (weeks)
        The number of weeks in the training window

    buff_window : int, default = 1 * 52 (weeks)
        The number of weeks to skip between the end of the training window
        and the start of the testing window.

    test_window : int, default = 1 * 52 (weeks)
        The number of weeks in the testing window

    Returns
    -------
    train : ndarray
        The training set indices for that split.

    test : ndarray
        The testing set indices for that split.
    """
    # check group
    if groups is None:
        raise ValueError('You have to pass in groups')
    if X is not None:
        check_consistent_length(force_array(X), force_array(groups))
    # get min/max date
    dates = np.sort(np.array(groups, dtype='datetime64[D]'))
    min_d = dates[0]
    max_d = dates[-1]
    # get test_fold
    if test_fold is None:
        test_start = max_d - np.timedelta64(test_window, 'W')
        test_fold = [test_start]
    else:  # if test_fold is NOT None
        if not isinstance(test_fold, (list, tuple, np.ndarray, pd.Index)):
            test_fold = [test_fold]
    # sort test_fold
    test_fold = np.sort(np.array(test_fold, dtype='datetime64[D]'))
    # check the last test fold
    last_test_start = test_fold[-1]
    if last_test_start >= max_d:
        raise ValueError('No testing data available for the last fold! '
                         'Please re-enter parameters!')
    # check the first training fold
    first_train_end = test_fold[0] - np.timedelta64(buff_window, 'W')
    if first_train_end <= min_d:
        raise ValueError('No trainning data available for the first fold! '
                         'Please re-enter parameters!')
    # generate index for rolling window folds
    for test_start in test_fold:
        # NOTE: there will be missing one day if it's in Leap Year
        # missing at 2004-12-31
        # get test_end
        test_end = test_start + np.timedelta64(test_window, 'W')
        test_end = np.min([max_d, test_end])
        # get train_end
        train_end = test_start - np.timedelta64(buff_window, 'W')
        # get train_start
        train_start = train_end - np.timedelta64(train_window, 'W')
        train_start = np.max([min_d, train_start])
        # get id
        train_start_idx = np.searchsorted(a=dates, v=train_start, side='left')
        train_end_idx = np.searchsorted(a=dates, v=train_end, side='right')
        test_start_idx = np.searchsorted(a=dates, v=test_start, side='left')
        test_end_idx = np.searchsorted(a=dates, v=test_end, side='right')
        yield (np.arange(start=train_start_idx, stop=train_end_idx, step=1),
               np.arange(start=test_start_idx, stop=test_end_idx, step=1))
