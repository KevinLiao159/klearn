"""
This module contains feature engineering computation/transformer,
All classes or functions should implement fit & transform methods.
All classes or functions should act like feature filters/selectors
All classes or functions only allow input data to be pandas
"""

# Authors: Kevin Liao

import numpy as np
import pandas as pd
import abc
import warnings
from scipy.stats import skew
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from gravity_learn.preprocessing import InfHandler
from gravity_learn.utils import check_is_fitted, force_array
from gravity_learn.logger import get_logger

logger = get_logger('preprocessing.features')


__all__ = ['FeatureSelector',
           'DiffMeanByTime',
           'BoLassoFeatSelector',
           'log_transformation',
           'cube_root_transformation',
           'NormalizeTransformer',
           'normalize_transformer',
           'BinningTransformer',
           'ReduceMultiCollinearFeatures']


class _BaseFeature(BaseEstimator, TransformerMixin):
    """
    Base class for feature selections, engineering,
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


class FeatureSelector(_BaseFeature):
    """Select columns of a pandas dataframe

    Parameters
    ----------
    feature_list: list or array, members are string
    """
    def __init__(self, feature_list=None):
        self.feature_list = feature_list

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self._type_check(X, y)
        if not self.is_dataframe:
            raise ValueError('X must be dataframe')
        return X[self.feature_list]


class DiffMeanByTime(_BaseFeature):
    """Calculate diff between each observation and mean of all per date

    Parameters
    ----------
    feature_list: list or array, members are string

    use_median: bool, default False, if True, diff against median instead
    """
    def __init__(self, feature_list, use_median=False, copy=True):
        self.feature_list = feature_list
        self.use_median = use_median
        self.copy = copy
        self.suffix = '__time_bucket_diff'

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self._type_check(X, y)
        if not self.is_dataframe:
            raise ValueError('X must be dataframe')
        if self.copy:
            X = X.copy()
        if self.use_median:
            X = X[self.feature_list].groupby(level='date')\
                    .transform(lambda x: x - x.median())
        else:
            X = X[self.feature_list].groupby(level='date')\
                    .transform(lambda x: x - x.mean())
        # rename columns
        X.rename(columns={col: col + self.suffix for col in self.feature_list})
        return X


class BoLassoFeatSelector(object):
    """
    This yields a length k list of column names which are the most
    significant according to lasso fits to n_sample bootstrapped
    samples.
    """
    @staticmethod
    def check_pos_int(x, arg_type):
        # Makes sure x is a positive integer, otherwise raises an error
        if int(x) != x:
            raise ValueError(
                'BoLassoFeatSelector must receive an integer for arg'
                ' {}'.format(arg_type))
        if x < 1:
            raise ValueError(
                'BoLassoFeatSelector must receive {} greater than 1'
                ' {}'.format(arg_type))

    def __init__(self, n_sample=10, sample_frac=0.5,
                 k=5, standardize=True, alpha=1):
        """
        Parameters
        ----------
        n_sample : number of bootstrapped samples to take

        sample_frac : the percent of the rows that should be sampled
            in each bootstrap

        k : the number of feats to ultimately be returned

        alpha : the alpha parameter that is fed into the lasso.

        Returns
        -------
        a list of the top k most relevant features.
        """
        self.check_pos_int(n_sample, 'n_sample')
        self.n_sample = n_sample
        if not (0 < sample_frac < 1):
            raise ValueError('BoLassoFeatSelector must recieve a sample_frac '
                             'between 0 and 1')
        self.sample_frac = sample_frac
        self.check_pos_int(k, 'k')
        self.k = k
        self.alpha = alpha
        self.standardize = standardize

    def pick_k(self, X, y):
        col_names = X.columns.values

        if X.shape[1] <= self.k:
            warnings.warn(
                'BoLassoFeatSelector is returning all columns because k is '
                '>= the number of columns'
            )
            return col_names

        index = list(range(X.shape[0]))
        if self.standardize:
            X = StandardScaler().fit_transform(X)
            X = pd.DataFrame(X, columns=col_names)

        sample_indices = [
            np.random.choice(
                index,
                round(self.sample_frac*X.shape[0]), replace=True
                ) for i in range(self.n_sample)
            ]
        col_scores = pd.DataFrame(
            dict(
                name=col_names,
                score=[0]*len(col_names)
                )
        )
        lass = Lasso(alpha=self.alpha)
        for s_i in sample_indices:
            lass.fit(X.loc[s_i, :], y[s_i])
            col_scores['score'] += lass.coef_

        col_scores['score'] = abs(col_scores['score'])
        col_scores = col_scores.sort_values(by='score', ascending=False)
        self.col_scores = col_scores
        top_feats = col_scores['name'].iloc[:self.k].tolist()
        top_scores = col_scores['score'].iloc[:self.k].tolist()
        if len(set(top_scores)) == 1:
            warnings.warn(
                'Warning: BoLassoFeatSelector is picking features that all '
                'have identical scores'
            )
        return top_feats


def log_transformation(x, k=1e-3, with_median=False):
    """
    This function does log transformation on a 1d vector.

    formula:
        x' = log(x/mean(x) + k), where k is a small constant (k<<1)

    NOTE: x is assumed that 99% of its data have positive values

    Parameters
    ----------
    x: 1d numpy array or a pandas series

    k: float, a scaler shift mean by log(k)

    with_median: bool, if True, then x' = log(x/median(x) + k)

    Returns
    -------
    x_log_transformed: 1d numpy array
    """
    # get mean
    if with_median:
        x_mean = np.nanmedian(x)
    else:
        x_mean = np.nanmean(x)
    # check the sign of mean
    if x_mean <= 0:
        raise ValueError('x is centered at nagative domain, '
                         'please do NOT log_transformation')
    # assume np.log handle negative, we just gives warnings
    neg_pct = (x <= 0).sum() / x.shape[0]
    if neg_pct > 0.1:
        logger.warning('RuntimeWarning: x has {:.2f}% of its data '
                       'less than zero and it would return nan '
                       ''.format(neg_pct))
        if neg_pct > 0.5:
            raise RuntimeError(
                'x has {:.2f}% of its data less than zero '
                ''.format(neg_pct))
    return np.log((x/x_mean) + k)


def cube_root_transformation(x):
    """
    This function does cube_root transformation on a 1d vector.

    formula:
        x' = x ** (1/3)

    Parameters
    ----------
    x: 1d numpy array or a pandas series

    Returns
    -------
    x_cbrt_transformed: 1d numpy array
    """
    return np.cbrt(x)


class NormalizeTransformer(_BaseFeature):
    """
    This class is responsible for special data transformation
    (eg. log, exp, or cbrt) based on the skewness of an underlying feature

    It would transform the only features that have skewness past a threshold
    and keep other features in original distribution

    NOTE: such type of data transformations on statistical distribution matters
        only when we are doing linear/regression models or correlation study.
        It is NOT neccessary useful for advanced model (eg. RandomForest)

    Parameters
    ----------
    skewness_threshold: float, if the absolute value of skewness of a feature \
        exceeds skewness_threshold, then special transformation is executed.
        One of following two things would happen:
            1. if skewness > skewness_threshold, then log / cbrt transformation
            2. if skewness < -1* skewness_threshold, then exp transformation

    abs_deviations_threshold: float, outliers with abs_deviations greater than\
        abs_deviations_threshold are ignored when computing skewness

    k: float, x' = log(x/mean(x) + k), where k is a small constant (k<<1)

    with_median: bool, if True, then x' = log(x/median(x) + k)
    """
    def __init__(self, skewness_threshold=1.0, abs_deviations_threshold=10.0,
                 k=1e-3, with_median=False):
        self.skewness_threshold = np.abs(skewness_threshold)
        self.abs_deviations_threshold = np.abs(abs_deviations_threshold)
        self.k = k
        self.with_median = with_median

    def _transform_per_feature(self, x):
        """
        Parameters
        ----------
        x: 1d array-like, vector, dtype, float

        Returns
        -------
        x_transformed: 1d array-like, vector, dtype, float
        """
        # check type
        if not isinstance(x[0], float):
            return x
        else:
            # calc mad
            median = np.nanmedian(x)
            mad = np.nanmedian(x - median)
            # fitler outliers
            hi = median + mad * self.abs_deviations_threshold
            lo = median - mad * self.abs_deviations_threshold
            hi_filter = x < hi
            lo_filter = x > lo
            # compute skewness
            skewness = skew(x[lo_filter & hi_filter])
            # transform
            if skewness > self.skewness_threshold:
                # check negative
                neg_pct = (x <= 0).sum() / x.shape[0]
                if neg_pct < 0.05:
                    x = log_transformation(
                        x=x,
                        k=self.k,
                        with_median=self.with_median
                    )
                else:
                    x = cube_root_transformation(x)
            elif skewness < (-1 * self.skewness_threshold):
                x = np.exp(x)
            return x

    def fit(self, X, y=None):
        """
        fit method do nothing, fit has to be called every
        """
        return self

    def transform(self, X, y=None):
        self._type_check(X, y)
        X = np.apply_along_axis(
            func1d=self._transform_per_feature,
            axis=0,
            arr=force_array(X)
        )
        if self.is_dataframe:
            X = pd.DataFrame(X, columns=self.df_cols, index=self.df_idx)
        return X


def normalize_transformer(X, skewness_threshold=1.0,
                          abs_deviations_threshold=10.0,
                          k=1e-3, with_median=False):
    """
    This function is a convenient function for above class NormalizeTransformer

    Parameters
    ----------
    X: numpy array or dataframe for data transformation

    skewness_threshold: float, if the absolute value of skewness of a feature \
        exceeds skewness_threshold, then special transformation is executed.
        One of following two things would happen:
            1. if skewness > skewness_threshold, then log / cbrt transformation
            2. if skewness < -1* skewness_threshold, then exp transformation

    abs_deviations_threshold: float, outliers with abs_deviations greater than\
        abs_deviations_threshold are ignored when computing skewness

    k: float, x' = log(x/mean(x) + k), where k is a small constant (k<<1)

    with_median: bool, if True, then x' = log(x/median(x) + k)

    Returns
    -------
    X_transformed:  numpy array or dataframe after data transformation
    """
    normal_trans = NormalizeTransformer(
        skewness_threshold=skewness_threshold,
        abs_deviations_threshold=abs_deviations_threshold,
        k=k,
        with_median=with_median
    )
    return normal_trans.fit_transform(X)


class BinningTransformer(_BaseFeature):
    """
    Label data (set feature values to 0, 1, ..etc) according to thresholds

    Values greater than first threshold map to 1, greater than second \
    threshold map to 2, and so on,  while values less than or equal to \
    the threshold map to 0.

    If only given one threshold, then it is a binarization transformation.

    Binarization is a common operation on text count data where the
    analyst can decide to only consider the presence or absence of a
    feature rather than a quantified number of occurrences for instance.

    It can also be used as a pre-processing step for estimators that
    consider boolean random variables (e.g. modelled using the Bernoulli
    distribution in a Bayesian setting).

    Parameters
    ----------
    col_thres_dict : dictionary
        eg. {
                'feature_one': [0, 5, 10],
                'feature_three': [-20, 10]
            }
        In above case, for 'feature_one' in dataframe,
        transformer will label zero for values that less than zero,
        label one for values that between zero and five, label two \
        for values that between five and ten, and label three for values \
        that greater than ten. Similar actions for 'feature_three'
    """
    def __init__(self, col_thres_dict):
        if not isinstance(col_thres_dict, dict):
            raise ValueError('Warning! col_thres_dict is NOT a dictionary')
        self.col_thres_dict = col_thres_dict

    @staticmethod
    def _label_encode(x, threshold):
        """
        Label encode for single vector according to thresholds

        Parameters
        ----------
        x: 1d numpy array

        threshold: list, a list of float or int

        Returns
        -------
        encoded vector, 1d numpy array with labels [0, 1, ...]
        """
        x_encoded = np.zeros(x.shape[0], dtype=np.int)
        for i, cutoff in enumerate(threshold):
            x_encoded[np.where(x > cutoff)[0]] = int(i + 1)
        return x_encoded

    def fit(self, X, y=None):
        """
        fit method do nothing, fit has to be called every time
        """
        return self

    def transform(self, X, y=None):
        self._type_check(X, y)
        X = force_array(X)
        # map col name to index if X is dataframe
        if self.is_dataframe:
            self.col_thres_dict = {
                np.where(self.df_cols == key)[0][0]: value
                for key, value in self.col_thres_dict.items()
            }
        # handle single vector X
        if X.ndim == 1:
            X = self._label_encode(X, self.col_thres_dict[0])
        else:  # for 2d array
            # iter thru columns
            for key, threshold in self.col_thres_dict.items():
                X[:, key] = self._label_encode(X[:, key], threshold)
        # finalized
        if self.is_dataframe:
            X = pd.DataFrame(X, columns=self.df_cols, index=self.df_idx)
        return X


class ReduceMultiCollinearFeatures(_BaseFeature):
    """
    This class is designed to automate features selection process,
    it is mainly responsible for removing collinear features

    Here is the implementation steps:
        1. scan distribution for each feature
        2. special transformations according to data skewness
        - x'=log(x+k) used for reducing right skewness of variables
        - Cube root used for reducing right skewness of variables with negative values   # noqa
        3. standardize data by sklean's StandardScaler()
        4. recursive feature pruning until VIF of each feature < threshold

    Parameters
    ----------
    vif_threshold: float, recursive pruning stops when vif of every feature < vif_threshold,
        vif = 1 / (1 - R^2). The Variance Inflation Factor (VIF) is a measure of colinearity \
        among predictor variables within a multiple regression. 
        It is calculated by taking the the ratio of the variance of all a given model's \
        betas divide by the variane of a single beta if it were fit alone.
        Vif between 5 to 10 is multicolinearity is likely present and you should \
        consider dropping the variable.

    transformation: str, must be one of ['auto', 'log', 'cube_root'],
        None: it do NOT implement any data transformation

        'auto': it will automatically chose either log or cube_root
                default is 'auto'

        # NOTE: for the time being, it won't support following manual options
        # 'log': x'=log(x/mean(x)+k), where k is a small constant (k<<1). 
        #     In this transformation, the mean x will be transformed to near x'=0 
        #     and k will function as a shape factor (small k will make x' more left-skewed, 
        #     larger k will make it less so).
        
        # 'cube_root' : Cube root has its own advantage. It can be applied to negative \
        #     values including zero

    skewness_threshold: float, if the absolute value of skewness of a feature \
        exceeds skewness_threshold, then special transformation is executed.
        One of following two things would happen:
            1. if skewness > skewness_threshold, then log / cbrt transformation
            2. if skewness < -1* skewness_threshold, then exp transformation

    abs_deviations_threshold: float, outliers with abs_deviations greater than\
        abs_deviations_threshold are ignored when computing skewness

    k: float, x' = log(x/mean(x) + k), where k is a small constant (k<<1)
    """
    def __init__(self, vif_threshold=5.0, transformation='auto',
                 skewness_threshold=1.0, abs_deviations_threshold=10.0,
                 k=1e-3):
        allowed_transformations = [None, 'auto']
        if transformation not in allowed_transformations:
            raise ValueError('transformation must be one of {} '
                             ''.format(allowed_transformations))
        self.vif_threshold = vif_threshold
        self.transformation = transformation
        self.skewness_threshold = skewness_threshold
        self.abs_deviations_threshold = abs_deviations_threshold
        self.k = k

    def fit(self, X, y=None):
        self._type_check(X, y)
        X = force_array(X)
        # data transformation and standardization
        if self.transformation == 'auto':
            transformer_pipe = Pipeline(
                [
                    ('inf_handler', InfHandler(strategy='max', refit=True)),
                    ('imputer', Imputer(
                        missing_values="NaN",
                        strategy='mean')),
                    ('normalize_transformer', NormalizeTransformer(
                        skewness_threshold=self.skewness_threshold,
                        abs_deviations_threshold=self.abs_deviations_threshold,
                        k=self.k)),
                    ('standard_scaler', StandardScaler())
                ]
            )
            X = transformer_pipe.fit_transform(X)
        # Initialization
        n_features = X.shape[1]
        support_ = np.ones(n_features, dtype=np.bool)
        features = np.arange(n_features)[support_]
        # Elimination
        # first pass
        vifs = np.array(
            [variance_inflation_factor(X, i) for i in range(n_features)])
        # recursive pruning
        while vifs.max() > self.vif_threshold:  # there is at least one feature
            # remaining features
            support_[features[np.argsort(vifs)[-1]]] = False
            features = np.arange(n_features)[support_]
            # get vifs for the remaining features
            vifs = np.array(
                [variance_inflation_factor(X[:, features], i)
                 for i in range(X[:, features].shape[1])])
        # set final attributes
        self.support_ = support_
        self.n_features_ = support_.sum()
        self.features = np.arange(n_features)[support_]
        self.vifs = vifs
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, 'features')
        self._type_check(X, y)
        X = force_array(X)[:, self.features]
        if self.is_dataframe:
            X = pd.DataFrame(
                X,
                columns=self.df_cols[self.support_],
                index=self.df_idx)
        return X
