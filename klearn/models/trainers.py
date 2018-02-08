"""trainers are responsible for training and saving models"""

# Authors: Kevin Liao
import os
import abc
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from gravity_learn.utils import (force_array,
                                 ensure_2d_array,
                                 check_consistent_length,
                                 fit_model,
                                 check_is_fitted,
                                 save_object)
from gravity_learn.logger import get_logger

logger = get_logger('models.trainers')


__all__ = ['GeneralTrainer']


class _BaseTrainer(object):
    """
    Base class for trainers, trainers implement train, save_predictions, \
    save_probas, save_models, and evaluate

    Use cases:
        1. Specifically for forecasting future data task, we can train and \
            evaluate models on a rolling window basis
        2. Save models for data in different rolling windows
        3. Save out-of-sample probas for further stacking technique (scalable)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, model,
                 models_location=None,
                 predictions_location=None,
                 probas_location=None,
                 cv=None,
                 scoring=None,
                 n_jobs=1,
                 verbose=1):
        """
        Parameters
        ----------
        model : instantiated machine learning model that implements fit and \
            predict

        models_location : directory path for saving trained models. If None,
            then trained model will NOT be saved. Default is None

        predictions_location : directory path for saving out-of-sampel \
            predictions. If None, then predictions will NOT be saved.
            Default is None

        predictions_location : directory path for saving out-of-sampel probas.
            If None, then probas will NOT be saved. Default is None

        cv : list of tuples of index for (train, test)
            eg. [
                    (array([0, 1, 2, 3, 4, 5, ...]),
                     array([10, 11, 12, 13, 14, 15, ...])),
                    (array([20, 21, 22, 23, 24, 25, ...]),
                     array([30, 31, 32, 33, 34, 35, ...])),
                ]
            If None, then trainer will train on entire data set.
            Saving predictions and probas will be disabled
            Default is None

        scoring : string, callable, list/tuple, dict or None, default: None
            A single string (see :ref:`scoring_parameter`) or a callable
            (see :ref:`scoring`) to evaluate the predictions on the test set.

            For evaluating multiple metrics, either give a list of (unique)
            strings or a dict with names as keys and callables as values.

            NOTE that when using custom scorers, each scorer should return a
            single value. Metric functions returning a list/array of values \
            can be wrapped into multiple scorers that return one value each.

            See :ref:`multimetric_grid_search` for an example.

            If None, the estimator's default scorer (if available) is used.

        n_jobs : integer, optional
            The number of CPUs to use to do the computation. -1 means
            'all CPUs'.

        verbose : integer, optional
            The verbosity level.
        """
        # make dir for pickling models
        if models_location is None:
            self.models_location = os.path.join(os.getcwd(), 'models')
        if predictions_location is None:
            self.predictions_location = os.path.join(os.getcwd(), 'predictions')           # noqa
        if probas_location is None:
            self.probas_location = os.path.join(os.getcwd(), 'probas')
        # init the rest
        self.model = deepcopy(model)
        if cv is None:
            logger.warning('If cv is None, '
                           'then it will train on entire data set. '
                           'predictions and probas will NOT be saved')
            self.predictions_location = None
            self.probas_location = None
        self.cv = cv
        # TODO: what is the best way to implement eval method?
        # self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    def train(self, X, y,
              save_models=True,
              save_predictions=True,
              save_probas=True):
        # train models
        self._train(X, y)
        # save object
        if save_models + save_predictions + save_probas > 0:
            self._save(
                X=X,
                y=y,
                cv=self.cv,
                save_models=save_models,
                save_predictions=save_predictions,
                save_probas=save_probas
            )
        return self

    def _train(self, X, y):
        # check X, y
        check_consistent_length(X, y)
        X = ensure_2d_array(X, axis=1)
        y = ensure_2d_array(y, axis=1)
        # get cv for None
        if self.cv is None:
            self.cv = [tuple((force_array(range(len(X))), ))]
        # get model dict for each fold
        model_dict = {i: deepcopy(self.model) for i in range(len(self.cv))}
        # parallel
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        func = delayed(fit_model)
        fitted_model_list = parallel(
            func(self.model_dict[i], X[self.cv[i][0]], y[self.cv[i][0]])
            for (i, model) in model_dict.items()
        )
        # update models
        fitted_model_list = iter(fitted_model_list)
        self.model_dict = {
            i: next(fitted_model_list)
            for i in range(len(self.cv))
        }
        if self.verbose > 0:
            logger.info('Training is done!')

    def _save(self, X, y,
              save_models=True,
              save_predictions=True,
              save_probas=True):
        # check fitted
        check_is_fitted(self, 'model_dict')
        # check X, y
        check_consistent_length(X, y)
        # lazy don't want to handle single axis tensor
        X = ensure_2d_array(X, axis=1)
        y = ensure_2d_array(y, axis=1)
        # check locations
        if self.models_location is None:
            save_models = False
        if self.predictions_location is None:
            save_predictions = False
        if self.probas_location is None:
            save_probas = False
        if save_models + save_predictions + save_probas == 0:
            logger.warning('Warning! Nothing gets saved. '
                           'Please check locations or cv')
        # save object
        pred_dfs = []
        proba_dfs = []
        for i, model in self.model_dict.items():
            # save model
            if save_models:
                self.save_model(model, name='model_{}'.format(i))
            if save_predictions:
                if hasattr(model, 'predict'):
                    pred_dfs.append(model.predict(X[self.cv[i][1]]))
                else:
                    logger.warning('Model does NOT implement predict')
            if save_probas:
                if hasattr(model, 'predict_proba'):
                    pred_dfs.append(model.predict_proba(X[self.cv[i][1]]))
                else:
                    logger.warning('Model does NOT implement predict_proba')
        if pred_dfs:
            self.pred_out_of_sample = np.vstack(pred_dfs)
            self.save_prediction(self.pred_out_of_sample)
        if proba_dfs:
            self.proba_out_of_sample = np.vstack(proba_dfs)
            self.save_proba(self.proba_out_of_sample)
        if self.verbose > 0:
            logger.info('Saving is done')

    def save_model(self, model, name='model'):
        filepath = os.path.join(self.models_location, '{}.pkl'.format(name))
        save_object(model, filepath)

    def save_prediction(self, pred, name='prediction'):
        filepath = os.path.join(self.predictions_location, '{}.pkl'.format(name))          # noqa
        save_object(pred, filepath)

    def save_proba(self, proba, name='proba'):
        filepath = os.path.join(self.probas_location, '{}.pkl'.format(name))          # noqa
        save_object(proba, filepath)

    @abc.abstractmethod
    def evaluate(self):
        pass


class GeneralTrainer(_BaseTrainer):
    """
    GeneralTrainer implement its own version of train and evaluate

    Use cases:
        1. It is designed for training models over pandas dataframe object
        2. Saved predictions be in dataframe format with original index
        3. Evaluation process can rely on pandas utility functions
    """
    def __init__(self, model,
                 models_location=None,
                 predictions_location=None,
                 probas_location=None,
                 cv=None,
                 scoring=None,
                 n_jobs=1,
                 verbose=1):
        super(GeneralTrainer, self).__init__(
            model=model,
            models_location=models_location,
            predictions_location=predictions_location,
            probas_location=probas_location,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose)

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
        else:  # X is NOT a dataframe
            raise
