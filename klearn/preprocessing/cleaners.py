"""
Cleaners for removing inf, outliers, or scale the data
All cleaners in this module should respect the form of input data
eg. input numpy array, it should return numpy array
    input pandas dataframe, it should return pandas dataframe
"""

# Authors: Kevin Liao

import numpy as np
import pandas as pd
import abc
from sklearn.base import BaseEstimator, TransformerMixin
from gravity_learn.utils import check_is_fitted
from gravity_learn.utils import force_array
from gravity_learn.logger import get_logger

logger = get_logger(__name__)

__all__ = ['IdentityScaler',
           'InfHandler',
           'GravityCleaner',
           'TukeyOutliers',
           'tukeyoutliers',
           'MADoutliers']


class _CleanerBase(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):
    """
    Base class for all cleaners,
    must implement transform method
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def transform(self, X):
        pass

    @abc.abstractmethod
    def fit(self, X, y=None):
        pass

    def _type_check(self, X, y=None):
        """
        Parameters
        ----------
        X: numpy array, or pandas dataframe
        """
        self.type_check = True
        self.is_dataframe = False
        if isinstance(X, (pd.DataFrame)):
            self.is_dataframe = True
            self.df_idx = X.index
            self.df_cols = X.columns.values


class IdentityScaler(_CleanerBase):
    """
    This is scaler that don't do anything just return the same data
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # check is dataframe
        self._type_check(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, 'type_check')
        return X


class GravityCleaner(_CleanerBase):
    """
    This is a V0 StatQuant Strategy preprocess. It replaces inf with
    nan and fill nan with -1.0E12

    Parameters
    ----------
    fill : float, a value that replaces all infs and nans
    """
    def __init__(self, fill=-1.0E12):
        self.fill = fill

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Returns
        ----------
        X: same type as X, with all finite values
        """
        self._type_check(X)
        # remove inf and fillna
        X = force_array(
            pd.DataFrame(force_array(X))
            .replace([-np.inf, np.inf], np.nan)
            .fillna(np.float32(self.fill))
        )
        if self.is_dataframe:
            X = pd.DataFrame(X, columns=self.df_cols, index=self.df_idx)
        return X


class InfHandler(_CleanerBase):
    """
    Handle inf in underlying data

    Parameters
    ----------
    strategy: string, ['nan'(default), 'max', 'mean', 'median']

        This parameter is required for handling infinite values
        'nan':
            Replace Inf with Nan.
        'max':
            Replace Inf with max value
        'mean':
            Replace Inf with mean of that column
        'median':
            Replace Inf with median of column

    refit: bool, if True, when its transform method gets called,
        it would call fit method again. This allows feature space
        drifts away. Default is False
    """
    def __init__(self, strategy='nan', refit=False):
        allowed_strategy = ['nan', 'max', 'mean', 'median']
        if strategy.lower() not in allowed_strategy:
            raise Exception(
                'strategy must be one of {}'.format(allowed_strategy))
        self.strategy = strategy
        self.refit = refit

    def _reset(self):
        """Reset internal data-dependent state if necessary.
        __init__ parameters are not touched.
        """
        if hasattr(self, 'type_check'):
            del self.type_check
            del self.is_dataframe
            del self.max
            del self.min
            del self.mean
            del self.median

    def _replace(self, x):
        # HACK: super hack!!!
        x_max = x[-4]
        x_min = x[-3]
        x_mean = x[-2]
        x_median = x[-1]
        # this has to be last step
        x = x[:-4]
        try:
            is_float = float(x[0])
            is_float = True
        except ValueError:
            is_float = False

        if is_float:
            x = np.asarray(x, dtype='float32')
            inf_filter = np.isinf(x)
            posinf_filter = np.isposinf(x)
            neginf_filter = np.isneginf(x)
            if sum(inf_filter):
                if self.strategy.lower() == 'nan':
                    x[inf_filter] = np.nan

                elif self.strategy.lower() == 'max':
                        x[posinf_filter] = x_max
                        x[neginf_filter] = x_min

                elif self.strategy.lower() == 'mean':
                        x[inf_filter] = x_mean

                elif self.strategy.lower() == 'median':
                        x[inf_filter] = x_median
        return x

    def fit(self, X, y=None):
        """
        Compute inf-free, nan-free, max, min, mean, and median
        for each column

        Parameters
        ----------
        X: array-like, it only allows int, float, and bool
        """
        # Reset internal state before fitting
        self._reset()
        # make X inf-free
        X = force_array(
            pd.DataFrame(force_array(X)).replace([-np.inf, np.inf], np.nan)
        )
        # HACK: for dealing with bool
        X = np.asarray(X, dtype='float32')
        # compute inf-free, nan-free, max, min, mean, and median for cols
        self.max = np.nanmax(a=force_array(X), axis=0)
        self.min = np.nanmin(a=force_array(X), axis=0)
        self.mean = np.nanmean(a=force_array(X), axis=0)
        self.median = np.nanmedian(a=force_array(X), axis=0)
        return self

    def _transform(self, X):
        """
        Returns
        ----------
        X: same type as X, with all finite values
        """
        self._type_check(X)
        # HACK: v stack max, min, mean, median
        try:
            X = np.vstack(
                [
                    force_array(X),
                    self.max,
                    self.min,
                    self.mean,
                    self.median
                ]
            )
        except ValueError:
            X = np.hstack(
                [
                    force_array(X),
                    self.max,
                    self.min,
                    self.mean,
                    self.median
                ]
            )
        X = np.apply_along_axis(func1d=self._replace, axis=0, arr=X)
        if self.is_dataframe:
            X = pd.DataFrame(X, columns=self.df_cols, index=self.df_idx)
        return X

    def transform(self, X):
        if self.refit:
            return self.fit(X)._transform(X)
        else:  # not refit
            return self._transform(X)


class TukeyOutliers(_CleanerBase):
    # NOTE: This is a univariate method
    """
    Replace outliers of each column by a percentile value of that column

    Parameters
    ----------
    lo: lower bound for outlier replacement values

    hi: upper bound for outlier replacement values

    refit: bool, if True, when its transform method gets called,
        it would call fit method again. This allows feature space
        drift away. Default is False
    """
    def __init__(self, lo=5, hi=95, refit=False):
        self.lo = lo
        self.hi = hi
        self.refit = refit

    def _reset(self):
        """Reset internal data-dependent state if necessary.
        __init__ parameters are not touched.
        """
        if hasattr(self, 'type_check'):
            del self.type_check
            del self.is_dataframe
            del self.median
            del self.outliers_thres
            del self.hi_replacement
            del self.lo_replacement

    @staticmethod
    def _get_outliers_thres(x):
        outliers_thres = 1.5*(np.percentile(x, 75)-np.percentile(x, 25))
        return outliers_thres

    def _replace(self, x):
        # HACK: super hack!!!
        median = x[-4]
        threshole = x[-3]
        hi = x[-2]
        lo = x[-1]
        # this has to be last step
        x = x[:-4]
        top_outlier_filter = (x - median) > threshole
        bottom_outlier_filter = (median - x) > threshole
        x[top_outlier_filter] = hi
        x[bottom_outlier_filter] = lo
        return x

    def fit(self, X, y=None):
        """
        Compute median, top, and bottom quantile replacement value
        for each column
        """
        # Reset internal state before fitting
        self._reset()
        # compute median, outliers threshole, hi, lo
        self.median = np.median(a=force_array(X), axis=0)
        self.outliers_thres = np.apply_along_axis(
            func1d=self._get_outliers_thres,
            axis=0,
            arr=force_array(X)
        )
        self.hi_replacement = np.percentile(a=force_array(X), q=self.hi, axis=0)    # noqa
        self.lo_replacement = np.percentile(a=force_array(X), q=self.lo, axis=0)    # noqa
        return self

    def _transform(self, X):
        """
        Returns
        -------
        X:  array-like, type and shape are preserved
            after truncating outliers and replacing them with quantile values
        """
        self._type_check(X)
        # HACK: v stack median, threshole, hi, lo
        try:
            X = np.vstack(
                [
                    force_array(X),
                    self.median,
                    self.outliers_thres,
                    self.hi_replacement,
                    self.lo_replacement
                ]
            )
        except ValueError:
            X = np.hstack(
                [
                    force_array(X),
                    self.median,
                    self.outliers_thres,
                    self.hi_replacement,
                    self.lo_replacement
                ]
            )
        X = np.apply_along_axis(func1d=self._replace, axis=0, arr=X)
        if self.is_dataframe:
            X = pd.DataFrame(X, columns=self.df_cols, index=self.df_idx)
        return X

    def transform(self, X):
        if self.refit:
            return self.fit(X)._transform(X)
        else:  # not refit
            return self._transform(X)


def tukeyoutliers(X, y=None, lo=5, hi=95):
    # NOTE: This is a univariate method
    """
    Replace outliers of each column by a percentile value of that column

    Parameters
    ----------
    X: numpy array, or dataframe

    y: 1-d array-like

    lo: lower bound for outlier replacement values

    hi: upper bound for outlier replacement values

    Returns
    -------
    X:  array-like, type and shape are preserved
        after truncating outliers and replacing them with quantile values
    """
    tukey = TukeyOutliers(lo=lo, hi=hi)
    return tukey.fit_transform(X, y)


class MADoutliers(_CleanerBase):
    # NOTE: This is a univariate method
    """
    This is an sklearn preprocessor that squishes column values
    such that their max and min is within a certain distance from
    their median. That distance is a user-specified (with argument
    abs_deviations) multiple of median absolute distances
    from the median.

    Parameters
    ----------
    abs_deviations: float, indicates how far away is replacement value
        from median, abs_deviations of 1.5 is about 1 std away

    refit: bool, if True, when its transform method gets called,
        it would call fit method again. This allows feature space
        drift away. Default is False
    """
    def __init__(self, abs_deviations=1.5, refit=False):
        self.abs_deviations = abs_deviations
        self.refit = refit

    def _reset(self):
        """Reset internal data-dependent state if necessary.
        __init__ parameters are not touched.
        """
        if hasattr(self, 'type_check'):
            del self.type_check
            del self.is_dataframe
            del self.median
            del self.mad

    @staticmethod
    def _calc_median_abs_deviation(x):
        """
        This calculates the median absolute distance each point has
        from the median.

        Input:
        ------
        x : 1d array-like, vector, dtype, float

        Returns:
        --------
        mad: median distance from the median
        median: median of x
        """
        nonan_filter = ~np.isnan(x)
        x_nonan = x[nonan_filter]
        median = np.median(x_nonan)
        mad = np.median([abs(e - median) for e in x_nonan])
        return median, mad

    def _squish(self, x):
        """
        This squishes the values of x to be within an abs_deviations number of
        mads from the median.

        Input:
        ------
        x : 1d array-like, vector

        Returns:
        --------
        x : 1d array-like, vector, after being squished
        """
        # HACK: super hack!!!
        x_median = x[-2]
        x_mad = x[-1]
        # this has to be last step
        x = x[:-2]
        if isinstance(x[0], np.float):
            x = np.asarray(x, dtype='float32')
            hi = x_median + x_mad * self.abs_deviations
            lo = x_median - x_mad * self.abs_deviations
            hi_filter = x > hi
            lo_filter = x < lo
            x[hi_filter] = hi
            x[lo_filter] = lo
        return x

    def fit(self, X, y=None):
        """
        Compute median and mad for each column. Assuming data is
        inf-free
        """
        # Reset internal state before fitting
        self._reset()
        # compute median, outliers threshole, hi, lo
        self.median = np.apply_along_axis(
            func1d=self._calc_median_abs_deviation,
            axis=0,
            arr=force_array(X)
        )[0]
        self.mad = np.apply_along_axis(
            func1d=self._calc_median_abs_deviation,
            axis=0,
            arr=force_array(X)
        )[1]
        return self

    def _transform(self, X):
        """
        Returns
        ----------
        X: same type as X, with squished values
        """
        self._type_check(X)
        # HACK: v stack median, mad
        try:
            X = np.vstack(
                [
                    force_array(X),
                    self.median,
                    self.mad
                ]
            )
        except ValueError:
            X = np.hstack(
                [
                    force_array(X),
                    self.median,
                    self.mad
                ]
            )
        X = np.apply_along_axis(func1d=self._squish, axis=0, arr=X)
        if self.is_dataframe:
            X = pd.DataFrame(X, columns=self.df_cols, index=self.df_idx)
        return X

    def transform(self, X):
        if self.refit:
            return self.fit(X)._transform(X)
        else:  # not refit
            return self._transform(X)
