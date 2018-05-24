"""trainers are responsible for training and saving models"""

# Authors: Kevin Liao
import os
import abc
from copy import deepcopy
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals.joblib import Parallel, delayed
from gravity_learn.model_selection import top_bottom_accuracy_score
from gravity_learn.utils import (check_gravity_index,
                                 force_array,
                                 ensure_2d_array,
                                 check_consistent_length,
                                 fit_model,
                                 check_has_set_attr,
                                 save_object,
                                 load_object,
                                 check_cv)
from gravity_learn.logger import get_logger

logger = get_logger(__name__)


__all__ = ['Trainer',
           'GeneralTrainer',
           'CalibratedTrainer',
           'GravityTrainer',
           'gravity_evaluate']


class _BaseTrainer(metaclass=abc.ABCMeta):
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
                 save_location=None,
                 cv=None,
                 n_jobs=1,
                 verbose=1):
        """
        Parameters
        ----------
        model : instantiated machine learning model that implements fit and \
            predict

        save_location : directory path for saving trained models,
            out-of-sample predictions and out-of-sampel probas. If None,
            it won't write anything to disk

        cv : list of tuples of index for (train, test)
            eg. [
                    (array([0, 1, 2, 3, 4, 5, ...]),
                     array([10, 11, 12, 13, 14, 15, ...])),
                    (array([20, 21, 22, 23, 24, 25, ...]),
                     array([30, 31, 32, 33, 34, 35, ...])),
                ]
            If None, then trainer will use sklearn default cv

        n_jobs : integer, optional
            The number of CPUs to use to do the computation. -1 means
            'all CPUs'.

        verbose : integer, optional
            The verbosity level.
        """
        # set dir for pickling models
        self.save_location = save_location
        if self.save_location is not None:
            self.models_location = os.path.join(self.save_location, 'models')
            self.predictions_location = os.path.join(self.save_location, 'predictions')           # noqa
            self.probas_location = os.path.join(self.save_location, 'probas')
        elif self.save_location is None:
            self.models_location = None
            self.predictions_location = None
            self.probas_location = None

        # init the cv
        if cv is None:
            logger.warning('If cv is None, '
                           'then it will use the default 3-fold '
                           'cross-validation')
            self.predictions_location = None
            self.probas_location = None
        self.cv = check_cv(cv)
        # get model dict for each fold
        self.model_dict = \
            {i: deepcopy(model) for i in range(self.cv.get_n_splits())}
        # init the rest
        self.n_jobs = n_jobs
        self.verbose = verbose

    def save_model(self, model, name='model'):
        check_has_set_attr(self, 'models_location')
        pathlib.Path(self.models_location).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(self.models_location, '{}.pkl'.format(name))
        save_object(model, filepath)

    def save_prediction(self, pred, name='prediction'):
        check_has_set_attr(self, 'predictions_location')
        pathlib.Path(self.predictions_location).mkdir(parents=True, exist_ok=True)       # noqa
        filepath = os.path.join(self.predictions_location, '{}.pkl'.format(name))        # noqa
        save_object(pred, filepath)

    def save_proba(self, proba, name='proba'):
        check_has_set_attr(self, 'probas_location')
        pathlib.Path(self.probas_location).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(self.probas_location, '{}.pkl'.format(name))             # noqa
        save_object(proba, filepath)

    @property
    def get_model_dict(self):
        # check fitted
        check_has_set_attr(self, 'model_dict')
        return self.model_dict

    @property
    def get_trained_model_dict(self):
        # check fitted
        check_has_set_attr(self, 'is_trained')
        return self.model_dict

    @property
    def get_out_of_sample_predictions(self):
        # check fitted
        check_has_set_attr(self, 'pred_out_of_sample')
        return self.pred_out_of_sample

    @property
    def get_out_of_sample_probas(self):
        # check fitted
        check_has_set_attr(self, 'proba_out_of_sample')
        return self.proba_out_of_sample

    @property
    def get_predictions_dict(self):
        # check fitted
        check_has_set_attr(self, 'preds_dict')
        return self.preds_dict

    @property
    def get_probas_dict(self):
        # check fitted
        check_has_set_attr(self, 'probas_dict')
        return self.probas_dict

    def train(self, X, y,
              save_models=True,
              save_predictions=True,
              save_probas=True):
        # train models
        self._train(X, y)
        # set attribute
        self.is_trained = True
        # save object
        self._save(
            X=X,
            y=y,
            save_models=save_models,
            save_predictions=save_predictions,
            save_probas=save_probas
        )
        return self

    @abc.abstractmethod
    def _train(self, X, y):
        pass

    @abc.abstractmethod
    def _save(self, X, y,
              save_models=True,
              save_predictions=True,
              save_probas=True):
        pass

    def evaluate(self, X=None, y=None,
                 kind='prediction',
                 scoring=None,
                 aggregator=None,
                 **score_kwargs):
        """
        This is a convenient method for quick evaluating out-of-sample scores

        Parameters
        ----------
        X : X is NOT required

        y : y has to be the same y passed in its train method

        kind : str, one of ['prediction', 'proba']. If 'prediction' is chosen,
            then it will score prediction against out of sample targets
            If 'proba' is chosen, then it will score proba against
            out of sample targets

        scoring : dictionary with {metrics name: metrics callable}
            eg. {'accuracy': sklearn.metrics.accuracy_score}
            Default is accuracy

        aggregator: a function or a callable, to aggregate a vector

        **score_kwargs : this is passed to metrics callable

        Returns
        -------
        score_dict : a dictionary of score
            eg. {
                    'accuracy': [0.84, 0.92, 0.86, 0.78],
                    'roc_auc': [0.72, 0.77, 0.73, 0.69]
                }
        """
        allowed_kind = ['prediction', 'proba']
        if kind not in allowed_kind:
            raise ValueError('kind must be one of {}'.format(allowed_kind))
        if kind == 'prediction':
            check_has_set_attr(self, 'preds_dict')
            y_hat_dict = self.preds_dict
        else:  # kind == 'proba'
            check_has_set_attr(self, 'probas_dict')
            y_hat_dict = self.probas_dict
            for i, y_probas in y_hat_dict.items():
                if np.dim(y_probas) == 2:
                    y_hat_dict[i] = y_probas[:, -1]
        # check y
        if y is None:
            raise ValueError('You must pass in y')
        else:
            y = force_array(y)
        # check scoring
        if scoring is None:
            scoring = {'accuracy': accuracy_score}
        # score out of sample
        score_dict = {}
        for name, score in scoring.items():
            # get scores for every folds
            scores_list = [
                score(y[self.cv[i][1]], y_hat_dict[i], **score_kwargs)
                for i in range(len(self.cv))
            ]
            # save scores with score name in score_dict
            score_dict = {
                **score_dict,
                **{name: scores_list}
            }
        # aggregator
        if aggregator:
            score_dict = {
                name: aggregator(scores)
                for (name, scores) in score_dict.items()
            }
        return score_dict


class Trainer(_BaseTrainer):
    """
    Trainer implement its own version of _train, and _save methods

    Use cases:
        1. It's a prototype trainer and it only supports numpy
        2. Not for pandas dataframe, models have no notion about index
            or columns that come with pandas dataframe
    """
    def __init__(self, model,
                 save_location=None,
                 cv=None,
                 n_jobs=1,
                 verbose=1):
        super(Trainer, self).__init__(
            model=model,
            save_location=save_location,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose)

    def _train(self, X, y):
        # check X, y
        check_consistent_length(X, y)
        X = ensure_2d_array(X, axis=1)
        y = ensure_2d_array(y, axis=1)
        # check cv
        self.cv = list(self.cv.split(X, y))
        # parallel
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        func = delayed(fit_model)
        fitted_model_list = parallel(
            func(model, X[self.cv[i][0]], y[self.cv[i][0]])
            for (i, model) in self.model_dict.items()
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
        check_has_set_attr(self, 'is_trained')
        # check X, y
        check_consistent_length(X, y)
        # lazy don't want to handle single axis tensor
        X = ensure_2d_array(X, axis=1)
        y = ensure_2d_array(y, axis=1)
        # check locations
        if self.save_location is None:
            logger.warning('Warning! Nothing gets saved. '
                           'Please reset save_location '
                           'if you want to write results to disk')
        # save object
        self.preds_dict = {}
        self.probas_dict = {}
        for i, model in self.model_dict.items():
            # save model
            if self.models_location and save_models:
                self.save_model(model, name='model_{}'.format(i))
            # predict
            if hasattr(model, 'predict'):
                self.preds_dict = {
                    **self.preds_dict,
                    **{i: model.predict(X[self.cv[i][1]])}
                }
            else:
                logger.warning('Model does NOT implement predict')
            # predict_proba
            if hasattr(model, 'predict_proba'):
                self.probas_dict = {
                    **self.probas_dict,
                    **{i: model.predict_proba(X[self.cv[i][1]])}
                }
            else:
                logger.warning('Model does NOT implement predict_proba')
        # collect data
        if self.preds_dict:
            preds_list = list(self.preds_dict.values())
            self.pred_out_of_sample = np.vstack(preds_list)
            # save pred
            if self.predictions_location and save_predictions:
                self.save_prediction(self.pred_out_of_sample)
        if self.probas_dict:
            probas_list = list(self.probas_dict.values())
            self.proba_out_of_sample = np.vstack(probas_list)
            # save probas
            if self.probas_location and save_probas:
                self.save_proba(self.proba_out_of_sample)
        if self.verbose > 0:
            logger.info('Saving is done')


class GeneralTrainer(_BaseTrainer):
    """
    General Trainer implements its own version of _train, _save


    Use cases:
        1. It's a general version of trainer and it supports numpy and
            pandas dataframe
        2. It supports dataframe utility functions
        3. It saves predictions and probas in dataframe format
    """
    def __init__(self, model,
                 save_location=None,
                 cv=None,
                 n_jobs=1,
                 verbose=1):
        super(GeneralTrainer, self).__init__(
            model=model,
            save_location=save_location,
            cv=cv,
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

    def _train(self, X, y):
        # check X, y
        check_consistent_length(X, y)
        self._type_check(X, y)
        if not self.is_dataframe:
            if not isinstance(X, (pd.DataFrame, pd.Series)):
                X = ensure_2d_array(X, axis=1)
                X = pd.DataFrame(X)
            if not isinstance(y, (pd.DataFrame, pd.Series)):
                y = ensure_2d_array(y, axis=1)
                y = pd.DataFrame(y)
        # check cv
        self.cv = list(self.cv.split(X, y))
        # parallel
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        func = delayed(fit_model)
        fitted_model_list = parallel(
            func(
                model,
                X.iloc[self.cv[i][0]],
                y.iloc[self.cv[i][0]])
            for (i, model) in self.model_dict.items()
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
        check_has_set_attr(self, 'is_trained')
        # check X, y
        check_consistent_length(X, y)
        if not self.is_dataframe:
            if not isinstance(X, (pd.DataFrame, pd.Series)):
                X = ensure_2d_array(X, axis=1)
                X = pd.DataFrame(X)
            if not isinstance(y, (pd.DataFrame, pd.Series)):
                y = ensure_2d_array(y, axis=1)
                y = pd.DataFrame(y)
        # check locations
        if self.save_location is None:
            logger.warning('Warning! Nothing gets saved. '
                           'Please reset save_location '
                           'if you want to write results to disk')
        # save object
        self.preds_dict = {}
        self.probas_dict = {}
        for i, model in self.model_dict.items():
            # save model
            if save_models:
                self.save_model(model, name='model_{}'.format(i))
            # pred
            if hasattr(model, 'predict'):
                self.preds_dict = {
                    **self.preds_dict,
                    **{
                        i: pd.DataFrame(
                            model.predict(X.iloc[self.cv[i][1]]),
                            index=X.iloc[self.cv[i][1]].index
                        )
                    }
                }
            else:
                logger.warning('Model does NOT implement predict')
            # probas
            if hasattr(model, 'predict_proba'):
                self.probas_dict = {
                    **self.probas_dict,
                    **{
                        i: pd.DataFrame(
                            model.predict_proba(X.iloc[self.cv[i][1]]),
                            index=X.iloc[self.cv[i][1]].index
                        )
                    }
                }
            else:
                logger.warning('Model does NOT implement predict_proba')

        if self.preds_dict:
            preds_list = list(self.preds_dict.values())
            self.pred_out_of_sample = \
                pd.concat(preds_list, verify_integrity=True).sort_index()
            # save pred
            if self.predictions_location and save_predictions:
                self.save_prediction(self.pred_out_of_sample)
        if self.probas_dict:
            probas_list = list(self.probas_dict.values())
            self.proba_out_of_sample = \
                pd.concat(probas_list, verify_integrity=True).sort_index()
            # save probas
            if self.probas_location and save_probas:
                self.save_proba(self.proba_out_of_sample)
        if self.verbose > 0:
            logger.info('Saving is done')


class CalibratedTrainer(GeneralTrainer):
    """
    Calibrated Trainer implements its own version of _train method
    for calibrated

    Use cases:
        1. It's a calibrated version of trainer and it supports only numpy
        2. It's designed for calibrating classifier estimators
        3. NOTE calibrated models do NOT support pandas dataframe

    New Parameters
    --------------
    calibrated_method : 'sigmoid' or 'isotonic'. The method to use for
        calibration. Can be 'sigmoid' which corresponds to Platt's method
        or 'isotonic' which is a non-parametric approach. It is not advised
        to use isotonic calibration with too few calibration samples (<<1000)
        since it tends to overfit. Use sigmoids (Platt's calibration)
        in this case.

    calibrated_cvs : list, a list of cv's for calibration. The number of cv's
        should be equal to the number of folds in cv. Default is None.
        If None, then to use sklean default 3-fold cross-validation
    """
    def __init__(self, model,
                 calibrated_method='sigmoid',
                 save_location=None,
                 cv=None,
                 calibrated_cvs=None,
                 n_jobs=1,
                 verbose=1):
        super(CalibratedTrainer, self).__init__(
            model=model,
            save_location=save_location,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose)
        # set calibrated_method
        allowed_method = ['sigmoid', 'isotonic']
        if calibrated_method not in allowed_method:
            raise ValueError('calibrated_method must be one of {} '
                             ''.format(allowed_method))
        self.calibrated_method = calibrated_method
        # calibrated_cvs
        if calibrated_cvs is None:
            calibrated_cvs = \
                [check_cv(3) for i in range(self.cv.get_n_splits())]
        elif not isinstance(calibrated_cvs, (list)):
            calibrated_cvs = [calibrated_cvs]
        if len(calibrated_cvs) != self.cv.get_n_splits():
            raise ValueError('calibrated_cvs has different length than cv')
        self.calibrated_cvs = calibrated_cvs

    def _train(self, X, y):
        # modified model_dict
        for (i, model) in self.model_dict.items():
            calibrated_model = CalibratedClassifierCV(
                base_estimator=deepcopy(model),
                method=self.calibrated_method,
                cv=self.calibrated_cvs[i]
            )
            self.model_dict[i] = deepcopy(calibrated_model)
        # train calibration models
        super(CalibratedTrainer, self)._train(X, y)


class GravityTrainer(GeneralTrainer):
    """
    Gravity Trainer implements its own version of evaluate method

    Use cases:
        It's used specifically for gravity research, which involved
        multi-level index ['date', 'tradingitemid']
        NOTE: this trainer is best if you use it for saving probas and
                scoring
    """
    def __init__(self, model,
                 save_location=None,
                 cv=None,
                 n_jobs=1,
                 verbose=1):
        super(GravityTrainer, self).__init__(
            model=model,
            save_location=save_location,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose)

    def set_cv(self, cv, X, y=None):
        # HACK: to get around the training steps
        self._type_check(X, y)
        self.cv = check_cv(cv)
        self.cv = list(self.cv.split(X, y))
        return self

    def set_model_dict(self, models_location):
        """
        convenient method for jaming fitted models back to object

        Parameter
        ---------
        models_location : directory of fitted models
        """
        # load paths
        models_location_list = [
            os.path.join(models_location, model_pkl)
            for model_pkl in os.listdir(models_location)
            if model_pkl.endswith('.pkl')
        ]
        # sort paths
        models_location_list.sort(key=os.path.getmtime)
        # queue in model_dict
        for i, model_pkl in enumerate(models_location_list):
            model = load_object(model_pkl)
            self.model_dict[i] = model
        # set is_trained
        # HACK: for saving new predictions
        self.is_trained = True
        return self

    def set_out_of_sample_probas(self, df_probas):
        """
        convenient method for puting probas in the object

        Parameter
        ---------
        df_probas: dataframe with gravity index
            NOTE: make sure it's column in order of \
                ['short_proba', 'long_proba']
        """
        self.proba_out_of_sample = df_probas
        return self

    def evaluate(self, X=None, y=None,
                 level='date',
                 scoring=None,
                 aggregator=None,
                 **score_kwargs):
        """
        This is a convenient method for quick evaluating out-of-sample scores
        from gravity research. The score will be calculated on per level basis

        NOTE it's designed specifically for gravity research. It can be further
            refactored into new forms for other research

        NOTE it does NOT support 'kind', because it always assume using probas
            for scoring evaluations

        Parameters
        ----------
        X : X is NOT required

        y : y has to be the same y passed in its train method

        level : str, one of ['date', 'tradingitemid']

        scoring : dictionary with {metrics name: metrics callable}
            eg. {'accuracy': sklearn.metrics.accuracy_score}
            Default is top_bottom_accuracy_score

        aggregator: a function or a callable, to aggregate a vector

        **score_kwargs : this is passed to metrics callable

        Returns
        -------
        score_dict : a dictionary of score
            eg. {
                    'level': ['2007-01-05', '2007-01-12', '2007-01-19'],
                    'accuracy': [0.84, 0.92, 0.86],
                    'roc_auc': [0.72, 0.77, 0.73]
                }
        """
        allowed_level = ['date', 'tradingitemid']
        if level not in allowed_level:
            raise ValueError('level must be one of {}'.format(allowed_level))
        # check y
        if y is None:
            raise ValueError('You must pass in y')
        else:
            check_gravity_index(y)
        # join out of sample probas with out of sample groud truth
        check_has_set_attr(self, 'proba_out_of_sample')
        check_gravity_index(self.proba_out_of_sample)
        # check ndim of self.proba_out_of_sample
        if np.ndim(self.proba_out_of_sample) == 2:
            df_join = self.proba_out_of_sample.iloc[:, -1:].join(y, how='left')
        else:  # else if ndim is 1
            df_join = self.proba_out_of_sample.join(y, how='left')
        # check scoring
        if scoring is None:
            scoring = {'accuracy': top_bottom_accuracy_score}
        # score out of sample
        score_dict = \
            {level: df_join.index.get_level_values(level).unique().values}
        for name, score in scoring.items():
            # get scores for every point on level
            scores_list = df_join.groupby(level=level).apply(
                lambda df: score(
                    df.iloc[:, 1],
                    df.iloc[:, 0],
                    **score_kwargs)
            ).values
            # save scores with score name in score_dict
            score_dict = {
                **score_dict,
                **{name: scores_list}
            }
        # aggregator
        if aggregator:
            score_dict = {
                name: aggregator(scores)
                for (name, scores) in score_dict.items() if name != level
            }
        return score_dict


def gravity_evaluate(df_true, df_score, level='date', scoring=None,
                     aggregator=None, **score_kwargs):
        """
        This is a wrapper function for quick scoring
        NOTE it is specifically for gravity research

        Parameters
        ----------
        df_true : dataframe, gravity outcomes data with gravity index

        df_score : dataframe, gravity trainer out of sample probas with \
            gravity index

        level : str, one of ['date', 'tradingitemid']

        scoring : dictionary with {metrics name: metrics callable}
            eg. {'accuracy': sklearn.metrics.accuracy_score}
            Default is top_bottom_accuracy_score

        aggregator: a function or a callable, to aggregate a vector

        **score_kwargs : this is passed to metrics callable

        Returns
        -------
        score_dict : a dictionary of score
            eg. {
                    'level': ['2007-01-05', '2007-01-12', '2007-01-19'],
                    'accuracy': [0.84, 0.92, 0.86],
                    'roc_auc': [0.72, 0.77, 0.73]
                }
        """
        allowed_level = ['date', 'tradingitemid']
        if level not in allowed_level:
            raise ValueError('level must be one of {}'.format(allowed_level))
        # check input data
        check_consistent_length(df_true, df_score)
        check_gravity_index(df_true)
        check_gravity_index(df_score)
        # check ndim of df_score
        if np.ndim(df_score) == 2:
            df_join = df_score.iloc[:, -1:].join(df_true, how='left')
        else:  # else if ndim is 1
            df_join = df_score.join(df_true, how='left')
        # check scoring
        if scoring is None:
            scoring = {'accuracy': top_bottom_accuracy_score}
        # score out of sample
        score_dict = \
            {level: df_join.index.get_level_values(level).unique().values}
        for name, score in scoring.items():
            # get scores for every point on level
            scores_list = df_join.groupby(level=level).apply(
                lambda df: score(
                    df.iloc[:, 1],
                    df.iloc[:, 0],
                    **score_kwargs)
            ).values
            # save scores with score name in score_dict
            score_dict = {
                **score_dict,
                **{name: scores_list}
            }
        # aggregator
        if aggregator:
            score_dict = {
                name: aggregator(scores)
                for (name, scores) in score_dict.items() if name != level
            }
        return score_dict
