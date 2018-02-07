import pandas as pd
import numpy as np
from copy import deepcopy
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.externals.joblib import Parallel, delayed
from gravity_learn.utils import (force_array,
                                 check_cv,
                                 fit_model,
                                 check_is_fitted)


__all__ = ['EnsemblerClassifier',
           'QuickStackClassifier',
           'FullStackClassifier']


class EnsemblerClassifier(BaseEstimator, TransformerMixin):
    # TODO: require df? how to pass Yfactory in
    """
    This is a class to ensemble a set of given base models. The assumption
    is that those models are tuned (hyperparameters chosen). It works as
    follows.

    It accepts a dictionary of base models, the ensembler to combine them,
    a number of folds (to be used in the cross validation strategy) and
    a random state (to be used in the cross val strategy)

    The fit method:

        The ensemblers iterates through the base models, doing two things:

            - determining out of sample predictions (so n_folds fit-predict
            combinations). This is used for fitting the ensembler next.

            - fit the base model to the full data, which is used for the
            ensemblers predict method

        Notice this implies we have n_folds + 1 fits for each base model.

        With these out of sample predictions, it determines the parameters
        of the ensemblers.

    The predict method:

        Determines the predictions of each of the base models and then
        combines them with the fitted ensembler.
    """
    def __init__(self, base_models, ensembler_est, n_folds, random_state=0):
        """
        Parameters
        ----------
        base_models : a dictionary of model name/model pairs

        ensembler_est : an ensembler to combine the outputs of the base
        model

        n_folds : the number of folds to use when estimating the parameters
        of the ensemblers. Note: Ideally, n_folds should be high, because
        it makes the size of the base model fit for predictions and the
        base model fit for ensembler calibration more similar.

        random_state : the random state to use in the cross validaiton
        strategy
        """
        self.base_models = base_models
        self.ensembler_est = ensembler_est
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted_base_models = {}
        self.model_order = []
        warnings.warn('EnsemblerClassifier is deprecated, '
                      'please use FullStackClassifier instead',
                      DeprecationWarning)

    def fit(self, X, y):
        cv = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        base_predictions = {}

        for name, model in self.base_models.items():
            # This is for determining the ensembler parameters
            base_predictions[name] = cross_val_predict(
                model, X, y, cv=cv, method='predict_proba'
            )[:, 1]
            # This for the ensembler.predict method
            self.fitted_base_models[name] = model.fit(X, y)
            self.model_order.append(name)

        base_predictions = pd.DataFrame(
            base_predictions,
            index=X.index
        )[self.model_order]

        self.ensembler_est.fit(base_predictions, y)

        return self

    def predict_proba(self, X):
        base_predictions = {}

        for name, model in self.fitted_base_models.items():
            base_predictions[name] = model.predict_proba(X)[:, 1]

        base_predictions = pd.DataFrame(
            base_predictions,
            index=X.index
        )[self.model_order]

        return self.ensembler_est.predict_proba(base_predictions)


class QuickStackClassifier(BaseEstimator):
    """
    This class has a similar stacking structure but also is scalable,
    which means, it's objective to save computing run time on training
    in-sample-fold and outputing out-of-fold predictions for fitting ensembler

    Instead of doing K-fold training for each base model, it does only one-fold
    To have a good performance, it requires ensembler to be a simple model with
    only a few parameters to tune

    Parameters
    ----------
    base_models : list of (string, base_model) tuples. The first
        half of each tuple is the group name of the pipeline.

    ensembler : an ensembler to combine the outputs of the base models

    proba : bool, if True, model will implement predict_proba when it
            gets called

    full_train : bool, if True, its base models are trained with 100% data
        again and they are used for generating probas for new data
        Default is True

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
    def __init__(self, base_models, ensembler, proba=True,
                 full_train=True, cv=None, n_jobs=1, verbose=0):
        self.base_models = list(base_models)
        self.ensembler = ensembler
        self.proba = proba
        self.full_train = full_train
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        if self.cv is None:
            self.cv = KFold(n_splits=3, shuffle=True)
        warnings.warn('QuickStackClassifier is deprecated, '
                      'please use FullStackClassifier instead',
                      DeprecationWarning)

    def get_params(self, deep=True):
        return self.ensembler.get_params(deep=deep)

    def set_params(self, **params):
        return self.ensembler.set_params(**params)

    def _fit(self, X, y, *args, **kwargs):
        """
        private method to train n base models for last fold of cv
        """
        # get list of folds of indices
        self.last_fold = list(check_cv(self.cv).split(X, y))[-1]
        self.in_fold = self.last_fold[0]
        self.out_of_fold = self.last_fold[-1]
        # Paralellization
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        if isinstance(X, pd.DataFrame):
            if not isinstance(y, (pd.Series, pd.DataFrame)):
                y = pd.DataFrame(y)
            self.fitted_models = parallel(delayed(fit_model)(
                model=deepcopy(model),
                X=X.iloc[self.in_fold],
                y=y.iloc[self.in_fold],
                *args,
                **kwargs
                ) for (_, model) in self.base_models
            )
        else:  # X is not a dataframe
            self.fitted_models = parallel(delayed(fit_model)(
                model=deepcopy(model),
                X=X[self.in_fold],
                y=force_array(y)[self.in_fold],
                *args,
                **kwargs
                ) for (_, model) in self.base_models
            )
        # train model with full 100% data
        if self.full_train:
            self.full_fitted_models = parallel(delayed(fit_model)(
                model=deepcopy(model),
                X=X,
                y=y,
                *args,
                **kwargs
                ) for (_, model) in self.base_models
            )

    def fit(self, X, y, *args, **kwargs):
        """
        fit method is the method for fitting the ensembler and the trainning
        data is out-of-fold predictions from base_models
        """
        # call _fit
        self._fit(X, y, *args, **kwargs)
        # generate out-of-sample predictions and reserve same order!!
        proba_dfs = []
        if isinstance(X, pd.DataFrame):
            for i, model in enumerate(self.fitted_models):
                df_proba = pd.DataFrame(
                    {'proba_{}'.format(i): model.predict_proba(X.iloc[self.out_of_fold])[:, 1]},  # noqa
                    index=self.out_of_fold
                )
                proba_dfs.append(df_proba)
        else:  # X is not a dataframe
            for i, model in enumerate(self.fitted_models):
                df_proba = pd.DataFrame(
                    {'proba_{}'.format(i): model.predict_proba(X[self.out_of_fold])[:, 1]},       # noqa
                    index=self.out_of_fold
                )
                proba_dfs.append(df_proba)
        # horizontal concat dfs and revert to origin order
        df_out_of_fold_pred = pd.concat(proba_dfs, axis=1)
        # if need to convert to predict
        if not self.proba:
            df_out_of_fold_pred = df_out_of_fold_pred >= 0.5
        # Now train ensembler
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.DataFrame(y)
        self.ensembler.fit(
            X=df_out_of_fold_pred,
            y=y.iloc[self.out_of_fold],
            *args, **kwargs
        )
        # signal done fitting
        self.fitted = True
        return self

    def predict_proba(self, X, *args, **kwargs):
        check_is_fitted(self, 'fitted')
        # use full_trained model or not
        if self.full_train:
            base_models_list = self.full_fitted_models
        else:
            base_models_list = self.fitted_models
        # get pred from all base models
        proba_dfs = []
        for i, model in enumerate(base_models_list):
            df_proba = pd.DataFrame(
                {'proba_{}'.format(i): model.predict_proba(X)[:, 1]}
            )
            proba_dfs.append(df_proba)
        # horizontal concat P1 from all base models
        df_base_pred = pd.concat(proba_dfs, axis=1)
        if not self.proba:
            df_base_pred = df_base_pred >= 0.5
        # ensembler make predictions
        return self.ensembler.predict_proba(df_base_pred, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        df_proba = self.predict_proba(X, *args, **kwargs)[:, 1]
        df_pred = df_proba >= 0.5
        return force_array(df_pred)


def _base_model_cross_val(model, X, y, cv=None, proba=True, *args, **kwargs):
    """
    A private function that trains each base model for each fold
    and outputs fitted base models, its out-of-fold predictions,
    and array of y (in same order of out-of-fold predictions)
    for fitting ensembler

    Parameters
    ----------
    model : object, base model

    X : array-like, or dataframe

    y : array-like, or dataframe

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

    proba : bool, if True, model will implement predict_proba when it
            gets called

    Returns
    -------
    list of fitted model for each fold, Xt(out-of-fold pred),
        y(matched with Xt)
    """
    # get list of folds of indices
    all_folds = list(check_cv(cv).split(X, y))
    # check data type
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        X = pd.DataFrame(force_array(X))
    if not isinstance(y, (pd.DataFrame, pd.Series)):
        y = pd.DataFrame(force_array(y))
    # iterate each train-fold and fit base model
    fitted_models = [
        fit_model(
            model=deepcopy(model),
            X=X.iloc[train],
            y=y.iloc[train],
            *args,
            **kwargs
        ) for train, test in all_folds
    ]
    # generate out-of-sample predictions and reserve same order!!
    proba_dfs = []
    for i, (train, test) in enumerate(all_folds):
        df_proba = pd.DataFrame(
            {'proba': fitted_models[i].predict_proba(X.iloc[test])[:, 1]},       # noqa
            index=test
        )
        proba_dfs.append(df_proba)
    # concat dfs, sort index, and record index
    df_out_of_sample = pd.concat(proba_dfs).sort_index()
    idx = df_out_of_sample.index.values
    # get pred_out_of_sample
    pred_out_of_sample = \
        force_array(df_out_of_sample).reshape((len(df_out_of_sample), 1))
    # if need to convert to predict
    if not proba:
        pred_out_of_sample = pred_out_of_sample > 0.5
    # get y matched with pred_out_of_sample
    y_out_of_sample = y.iloc[idx]

    return fitted_models, pred_out_of_sample, y_out_of_sample


class FullStackClassifier(BaseEstimator):
    """
    This class is a full version of QuickStackClassifier, in other words,
    QuickStackClassifier is a sub-instance of FullStackClassifier

    Its objective is outputing out-of-fold predictions to fit ensembler
    Instead of passing Xt, y (keep same shape) to ensembler, this class is
    meant to allow Xt, y (modified shape due to specific CV strat) to ensembler

    Parameters
    ----------
    base_models : list of (string, base_model) tuples. The first
        half of each tuple is the group name of the pipeline.

    ensembler : an ensembler to combine the outputs of the base models

    proba : bool, if True, model will implement predict_proba when it
            gets called

    full_train : bool, if True, its base models are trained with 100% data
        again and they are used for generating probas for new data
        Default is True

    quick_stack : bool, if True, base models predict only on the last fold to
        output out-of-sample predictions for ensembler to fit.
        Default is False

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
    def __init__(self, base_models, ensembler, proba=True,
                 full_train=True, quick_stack=False,
                 cv=None, n_jobs=1, verbose=0):
        self.base_models = list(base_models)
        self.ensembler = ensembler
        self.proba = proba
        self.full_train = full_train
        self.quick_stack = quick_stack
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    def get_params(self, deep=True):
        return self.ensembler.get_params(deep=deep)

    def set_params(self, **params):
        return self.ensembler.set_params(**params)

    @property
    def get_fitted_models_(self):
        check_is_fitted(self, 'fitted')
        if self.full_train:
            fitted_models = self.full_fitted_models
        else:
            fitted_models = self.fitted_models
        return fitted_models

    @property
    def get_fitted_ensembler_(self):
        check_is_fitted(self, 'fitted')
        return self.ensembler

    def fit(self, X, y, *args, **kwargs):
        """
        fit method is the method for fitting the ensembler and the trainning
        data is out-of-fold predictions from base_models
        """
        # cv has to be deterministic
        cv = list(check_cv(self.cv).split(X, y))
        # check quick_stack
        if self.quick_stack:
            cv = [cv[-1]]
        # parallel iterating thru models to output out-of-fold pred
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        result = parallel(delayed(_base_model_cross_val)(
            model=deepcopy(model),
            X=X,
            y=y,
            cv=cv,
            proba=self.proba,
            *args, **kwargs
            ) for (_, model) in self.base_models
        )
        # post process
        fitted_models, pred_out_of_sample, y_out_of_sample = zip(*result)
        self.fitted_models = \
            [
                (self.base_models[i][0], models)
                for i, models in enumerate(fitted_models)
            ]
        # assume all y_out_of_sample are the same, which they should be
        y_out_of_sample = y_out_of_sample[0]
        # prepare out_of_sample to fit ensembler
        pred_out_of_sample = np.hstack(pred_out_of_sample)
        # Now train ensembler
        self.ensembler.fit(
            X=pred_out_of_sample,
            y=y_out_of_sample,
            *args, **kwargs
        )
        # check full_train
        if self.full_train:
            self.full_fitted_models = parallel(delayed(fit_model)(
                model=deepcopy(model),
                X=X,
                y=y,
                *args,
                **kwargs
                ) for (_, model) in self.base_models
            )
            # post process
            self.full_fitted_models = \
                [
                    (self.base_models[i][0], models)
                    for i, models in enumerate(self.full_fitted_models)
                ]
        # signal done fitting
        self.fitted = True
        return self

    def predict_proba(self, X, *args, **kwargs):
        check_is_fitted(self, 'fitted')
        # use full_trained model or not
        proba_dfs = []
        if self.full_train:
            for name, model in self.full_fitted_models:
                df_proba = pd.DataFrame(
                    {'proba_{}'.format(name): model.predict_proba(X)[:, 1]}
                )
                proba_dfs.append(df_proba)
        else:
            for name, models in self.fitted_models:
                avg_proba = np.average(
                    np.hstack(
                        [
                            model.predict_proba(X)[:, 1].reshape((len(X), 1))
                            for model in models
                        ]
                    ),
                    axis=1
                )
                df_proba = pd.DataFrame({'proba_{}'.format(name): avg_proba})
                proba_dfs.append(df_proba)
        # horizontal concat P1 from all base models
        df_base_pred = pd.concat(proba_dfs, axis=1)
        if not self.proba:
            df_base_pred = df_base_pred > 0.5
        # ensembler make predictions
        return self.ensembler.predict_proba(df_base_pred, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        df_proba = self.predict_proba(X, *args, **kwargs)[:, 1]
        df_pred = df_proba > 0.5
        return force_array(df_pred)
