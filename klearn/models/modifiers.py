# Authors: Kevin Liao <kevin.lwk.liao@gmail.com>

import numpy as np
import pandas as pd
from copy import deepcopy
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import KFold
from sklearn.externals.joblib import Parallel, delayed
from gravity_learn.utils import (force_array,
                                 check_cv,
                                 check_is_fitted)
from gravity_learn.logger import get_logger

logger = get_logger('models.modifiers')

__all__ = ['XGBClassifier2',
           'XGBRegressor2',
           'KDEClassifier',
           'ModelTransformer',
           'ModelBaseClassifier',
           'ModelTargetModifier',
           'AveragerClassifier']


class XGBClassifier2(XGBClassifier):
    """
    Wrapper of sklearn XGBoost classifier that implements transform,
    fit_transform, predict_transform methods

    Use Cases: pipeline, ensemble, stackings
    """
    def transform(self, X, y=None):
        return self.predict_proba(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict_proba(X)


class XGBRegressor2(XGBRegressor):
    """
    Wrapper of sklearn XGBoost regressor that implements transform,
    fit_transform methods

    Use Cases: pipeline, ensemble, stackings
    """
    def transform(self, X, y=None):
        return self.predict(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE

    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self

    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class ModelTransformer(BaseEstimator, TransformerMixin):
    """
    Use model predictions as transformer. Create a wrapper for
    any estimator so that it can implement fit and transform
    Parameters
    ----------
    model : object, estimator that is instantiated
    proba : bool, if True, model will implement predict_proba when it
            gets called
    """
    def __init__(self, model, proba=True):
        self.model = model
        self.proba = proba

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        return self.model.set_params(**params)

    def fit(self, X, y, *args, **kwargs):
        self.model.fit(X, y, *args, **kwargs)
        return self

    def transform(self, X, *args, **kwargs):
        if self.proba:
            y_hat = self.model.predict_proba(X, *args, **kwargs)[:, 1]
        else:
            y_hat = self.model.predict(X, *args, **kwargs)
        # reshape 1d array for horizontal merge in feature union
        y_hat = force_array(y_hat)
        y_hat = np.reshape(y_hat, newshape=(y_hat.shape[0], 1))
        return y_hat


def _fit_base_model(model, X, y, *args, **kwargs):
    # NOTE: temporary hack --- should clone model
    model.fit(X, y, *args, **kwargs)
    return model


class ModelBaseClassifier(BaseEstimator):
    """
    This is specific for base models in a context of stacking/ensemble.
    It creates a wrapper class for any base model so that it can implement
    fit_transform and transform methods. However, it won't have a fit method

    Parameters
    ----------
    model : object, estimator that is instantiated

    proba : bool, if True, model will implement predict_proba when it
            gets called

    full_train : bool, if True, base model is trained again with 100% data and
        it is used to transform new data. Default is True

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.
    """
    def __init__(self, model, proba=True, full_train=True,
                 cv=None, n_jobs=1, verbose=0):
        self.model = model
        self.proba = proba
        self.full_train = full_train
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        # shuffle introduces forward-looking bias
        # if self.cv is None:
        #     self.cv = KFold(n_splits=3, shuffle=True)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        return self.model.set_params(**params)

    def fit(self, X, y, *args, **kwargs):
        raise NotImplementedError

    def _fit(self, X, y, *args, **kwargs):
        """
        private method to train n base models for n folds of cv
        fit method should never get called
        """
        # get list of folds of indices
        self.folds = list(check_cv(self.cv).split(X, y))
        # Paralellization
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        if isinstance(X, pd.DataFrame):
            if not isinstance(y, (pd.Series, pd.DataFrame)):
                y = pd.DataFrame(y)
            self.fitted_models = parallel(delayed(_fit_base_model)(
                model=deepcopy(self.model),
                X=X.iloc[train],
                y=y.iloc[train],
                *args,
                **kwargs
                ) for train, test in self.folds
            )
        else:  # X is not a dataframe
            self.fitted_models = parallel(delayed(_fit_base_model)(
                model=deepcopy(self.model),
                X=X[train],
                y=force_array(y)[train],
                *args,
                **kwargs
                ) for train, test in self.folds
            )
        # train model with full 100% data
        if self.full_train:
            self.full_fitted_model = _fit_base_model(
                model=deepcopy(self.model),
                X=X,
                y=y,
                *args,
                **kwargs
            )

    def fit_transform(self, X, y, *args, **kwargs):
        """
        fit_transform method gets called when the ensemble is fitted to data
        It implements _fit to fit base models for different folds and output
        out-of-sample predictions
        """
        # call _fit
        self._fit(X, y, *args, **kwargs)
        # generate out-of-sample predictions and reserve same order!!
        proba_dfs = []
        if isinstance(X, pd.DataFrame):
            for i, (train, test) in enumerate(self.folds):
                df_proba = pd.DataFrame(
                    {'proba': self.fitted_models[i].predict_proba(X.iloc[test])[:, 1]},       # noqa
                    index=test
                )
                proba_dfs.append(df_proba)
        else:  # X is not a dataframe
            for i, (train, test) in enumerate(self.folds):
                df_proba = pd.DataFrame(
                    {'proba': self.fitted_models[i].predict_proba(X[test])[:, 1]},            # noqa
                    index=test
                )
                proba_dfs.append(df_proba)
        # concat dfs and revert to origin order
        df_pred = pd.concat(proba_dfs).sort_index()
        # get y_out_of_sample
        y_out_of_sample = force_array(df_pred).reshape((len(df_pred), 1))
        # if need to convert to predict
        if not self.proba:
            y_out_of_sample = y_out_of_sample > 0.5
        return y_out_of_sample

    def transform(self, X, *args, **kwargs):
        """
        transform method gets called when the ensemble is predicting
        It calls predict method on every model in self.fitted_models,
        then it will output average predictions from them
        """
        check_is_fitted(self, 'fitted_models')
        # output probas from full fitted model
        if self.full_train:
            pred = self.full_fitted_model\
                       .predict_proba(X)[:, 1].reshape((len(X), 1))
        else:  # get average probas from fitted models
            pred = np.average(
                np.hstack(
                    [
                        model.predict_proba(X)[:, 1].reshape((len(X), 1))
                        for model in self.fitted_models
                    ]
                ),
                axis=1
            ).reshape((len(X), 1))
        # if need to convert to predict
        if not self.proba:
            pred = pred > 0.5
        return pred


class ModelTargetModifier(BaseEstimator):
    """
    This model modifier will takes y_factory and do target feature
    engineering on y before fitting.

    Parameters
    ----------
    model : object, estimator that is instantiated

    y_factory : a callable that labels True/False for y according to \
                a given decision boundary. If None, then it does NOT
                do any feature engineering on y. Default is None.
    """
    def __init__(self, model, y_factory=None):
        self.model = model
        self.y_factory = y_factory

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        return self.model.set_params(**params)

    def fit(self, X, y, *args, **kwargs):
        if self.y_factory:
            y = self.y_factory(y)
        self.model.fit(X, y, *args, **kwargs)
        return self

    def predict(self, X, *args, **kwargs):
        return self.model.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        return self.model.predict_proba(X, *args, **kwargs)


class AveragerClassifier(BaseEstimator):
    """
    This model modifier will average probas from sub models.

    Parameters
    ----------
    weights : array_like, optional
        Each value in a contributes to the average according to its
        associated weight. The weights array should be 1-D
        If weights=None, then all data are assumed to have a weight equal to
        one.
    """
    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, X, y):
        return self

    def _predict_proba(self, X):
        # assuming X matrix only has P1
        n_samples = len(X)
        p1 = np.average(
            X,
            axis=1,
            weights=self.weights
        ).reshape((n_samples, 1))
        p0 = np.array([1]) - p1
        return np.hstack([p0, p1])

    def predict_proba(self, X):
        return self._predict_proba(X)

    def predict(self, X):
        p1 = self._predict_proba(X)[:, 1]
        thres = 0.5
        return p1 >= thres
