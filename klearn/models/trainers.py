"""trainers are responsible for training and saving models"""

# Authors: Kevin Liao <kevin.lwk.liao@gmail.com>
import os
import abc
from copy import deepcopy
import numpy as np
import pandas as pd
from gravity_learn.utils import save_object
from gravity_learn.logger import get_logger

logger = get_logger('models.trainers')


__all__ = ['']


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

        cv : list of tuples of index for (train, test)
            eg. [
                    (array([0, 1, 2, 3, 4, 5, ...]),
                     array([10, 11, 12, 13, 14, 15, ...])),
                    (array([20, 21, 22, 23, 24, 25, ...]),
                     array([30, 31, 32, 33, 34, 35, ...])),
                ]

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
                           'then it will train on entire data set')
        self.cv = cv
        # TODO: what is the best way to implement eval method?
        # self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    def train(self, X, y,
              save_models=False,
              save_predictions=False,
              save_probas=True):
        # TODO:
        pass

    def save_model(self, model, name='model'):
        filepath = os.path.join(self.models_location, '{}.pkl'.format(name))
        save_object(model, filepath)

    def save_prediction(self, pred, name='prediction'):
        filepath = os.path.join(self.predictions_location, '{}.pkl'.format(name))          # noqa
        save_object(pred, filepath)

    def save_proba(self, proba, name='proba'):
        filepath = os.path.join(self.probas_location, '{}.pkl'.format(name))          # noqa
        save_object(proba, filepath)

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
