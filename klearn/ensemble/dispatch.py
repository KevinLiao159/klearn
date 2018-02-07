import pandas as pd
import numpy as np
# from copy import deepcopy
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.joblib import Parallel, delayed
from gravity_learn.utils import (force_array,
                                 check_is_fitted,
                                 fit_model)


__all__ = ['ModelDispatch']


class ModelDispatch(_BaseComposition):
    """
    This class is responsible for dispatching models to different classes of
    data, class label of data is predicted by dispatcher (unsupervised \
    learning) or is labeled by y_train (supervised learning).
    Downstream models will consume its corresponding class of data and \
    implement fit and predict.

    Use cases:
        1. We want to have separate models for inliers and outliers
            (unsupervise)
        2. We want to have separate models for different groups of data with
            different values of y (supervise)

    Parameters
    ----------
    dispatcher : A classifier, could be unsupervised or supervised

    model_list : list of (string, base_model) tuples. The first
        half of each tuple is the group name of the model.

    supervise_cutoff : float, a cut-off, or a list of cut-offs to separate \
        two or more groups of data. If None, then assuming dispatch is an \
        unsupervise learning algo

        NOTE: assuming cut-offs are in ascending order

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.
    """
    def __init__(self, dispatcher, model_list, supervise_cutoff=None,
                 n_jobs=1, verbose=0):
        self.dispatcher = dispatcher
        self.model_list = list(model_list)
        self.supervise_cutoff = supervise_cutoff
        self.n_jobs = n_jobs
        self.verbose = verbose
        if supervise_cutoff is not None:
            if not isinstance(supervise_cutoff,
                              (list, tuple, np.ndarray, pd.Index)):
                supervise_cutoff = [supervise_cutoff]

    def get_params(self, deep=True):
        return self._get_params('model_list', deep=deep)

    def set_params(self, **kwargs):
        self._set_params('model_list', **kwargs)
        return self

    @property
    def get_model_list_(self):
        return self.model_list

    @property
    def get_model_dict_(self):
        check_is_fitted(self, 'model_dict')
        return self.model_dict

    def _fit_unsupervise(self, X, y=None, *args, **kwargs):
        self.dispatcher = self.dispatcher.fit(X, *args, **kwargs)
        self.group = self.dispatcher.predict(X)

    def _fit_supervise(self, X, y, *args, **kwargs):
        self.group = np.zeros(len(y))
        for i, cutoff in enumerate(self.supervise_cutoff):
            self.group[np.where(y > cutoff)[0]] = i + 1
        self.dispatcher = self.dispatcher.fit(X, self.group, *args, **kwargs)

    def fit(self, X, y=None, *args, **kwargs):
        # NOTE: let's say we respect dataframe
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            X = pd.DataFrame(force_array(X))
        if not isinstance(y, (pd.DataFrame, pd.Series)):
            y = pd.DataFrame(force_array(y))

        # First, fit dispatcher and get group
        if self.supervise_cutoff is None:
            self._fit_unsupervise(X, *args, **kwargs)
        else:  # supervise
            self._fit_supervise(X, y, *args, **kwargs)

        # Second, fit Label encoder
        self.le = LabelEncoder().fit(self.group)
        self.group = self.le.transform(self.group)
        self.unique_groups = np.unique(self.group)
        # Third, get a model dictionary for two class of data
        self.model_dict = \
            {
                group: self.model_list[i][-1]
                for i, group in enumerate(self.unique_groups)
            }
        # Nest, get list of index
        index_dict = \
            {
                group: np.where(self.group == group)[0]
                for group in self.unique_groups
            }
        # Paralization and fit downstream models
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        func = delayed(fit_model)
        fitted_model_list = parallel(
            func(self.model_dict[group], X.iloc[index], y.iloc[index])
            for (group, index) in index_dict.items()
        )
        # update models
        fitted_model_list = iter(fitted_model_list)
        self.model_dict = {
            group: next(fitted_model_list)
            for group in self.unique_groups
        }
        return self

    def predict(self, X):
        check_is_fitted(self, 'model_dict')
        # NOTE: let's say we respect dataframe
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            X = pd.DataFrame(force_array(X))

        # predict on dispatcher and get group
        group_new = self.dispatcher.predict(X)
        group_new = self.le.transform(group_new)
        index_dict = \
            {
                group: np.where(group_new == group)[0]
                for group in self.unique_groups
            }
        # predict by group
        pred_dfs = []
        for (group, index) in index_dict.items():
            if len(index):
                df_pred = pd.DataFrame(
                    self.model_dict[group].predict(X.iloc[index]),
                    index=index
                )
                pred_dfs.append(df_pred)
        # concat all predictions into one dataframe
        df_pred = pd.concat(pred_dfs)
        return force_array(df_pred.sort_index())

    def predict_proba(self, X):
        check_is_fitted(self, 'model_dict')
        # NOTE: let's say we respect dataframe
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            X = pd.DataFrame(force_array(X))

        # predict on dispatcher and get group
        group_new = self.dispatcher.predict(X)
        group_new = self.le.transform(group_new)
        index_dict = \
            {
                group: np.where(group_new == group)[0]
                for group in self.unique_groups
            }
        # predict by group
        proba_dfs = []
        for (group, index) in index_dict.items():
            if len(index):
                df_proba = pd.DataFrame(
                    self.model_dict[group].prodict_proba(X.iloc[index]),
                    index=index
                )
                proba_dfs.append(df_proba)
        # concat all prodictions into one dataframe
        df_proba = pd.concat(proba_dfs)
        return force_array(df_proba.sort_index())
