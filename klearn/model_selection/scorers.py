"""
Scores or metrics for model selection

A scorer object is a callable that can be passed to
:class:`sklearn.model_selection.GridSearchCV` or
:func:`sklearn.model_selection.cross_val_score` as the ``scoring``
parameter, to specify how a model should be evaluated.

The signature of the call is ``(estimator, X, y)`` where ``estimator``
is the model to be evaluated, ``X`` is the test data and ``y`` is the
ground truth labeling (or ``None`` in the case of unsupervised models).
"""

# Authors: Kevin Liao
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, log_loss)

from sklearn.metrics import make_scorer
from .metrics import (top_bottom_accuracy_score,
                      top_bottom_precision_score,
                      top_bottom_recall_score,
                      top_bottom_f1_score,
                      top_bottom_roc_auc_score,
                      top_bottom_log_loss)

from .metrics import (root_mean_squared_error,
                      mean_absolute_percentage_error)

from sklearn.model_selection import cross_validate


__all__ = ['cross_val_multiple_scores',
           'SCORERS',
           'accuracy_scorer',
           'precision_scorer',
           'recall_scorer',
           'f1_scorer',
           'roc_auc_scorer',
           'top_bottom_accuracy_scorer',
           'top_bottom_precision_scorer',
           'top_bottom_recall_scorer',
           'top_bottom_f1_scorer',
           'top_bottom_roc_auc_scorer',
           'neg_log_loss_scorer',
           'log_loss_scorer',
           'top_bottom_neg_log_loss_scorer',
           'top_bottom_log_loss_scorer',
           'root_mean_squared_error_scorer',
           'mean_absolute_percentage_error']


def cross_val_multiple_scores(estimator, X, y=None, groups=None, scoring=None,
                              cv=None, n_jobs=1, verbose=0, fit_params=None,
                              pre_dispatch='2*n_jobs', aggregator=None):
    """
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    aggregator: a function or a callable, to aggregate a vector

    Returns
    -------
    scores : dict of float arrays of shape=(n_splits,)
        Array of scores of the estimator for each run of the cross validation.

    Examples
    --------
    {
        'roc_auc': array([ 0.65427003,  0.65790116,  0.69799767]),
        'accuracy': array([ 0.70475276,  0.56329497,  0.6588])
    }
    """
    cv_results = cross_validate(estimator=estimator, X=X, y=y, groups=groups,
                                scoring=scoring, cv=cv,
                                return_train_score=False,
                                n_jobs=n_jobs, verbose=verbose,
                                fit_params=fit_params,
                                pre_dispatch=pre_dispatch)
    test_score_dict = {
        name.split('_', 1)[-1]: scores
        for (name, scores) in cv_results.items() if name.startswith('test')
    }
    # aggregator
    if aggregator:
        test_score_dict = {
            name: aggregator(scores)
            for (name, scores) in test_score_dict.items()
        }
    return test_score_dict


# Standard Classification Scores
accuracy_scorer = make_scorer(accuracy_score)
precision_scorer = make_scorer(precision_score)
recall_scorer = make_scorer(recall_score)
f1_scorer = make_scorer(f1_score)

# Score functions that need decision values
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)

# Score function for probabilistic classification
top_bottom_accuracy_scorer = make_scorer(top_bottom_accuracy_score,
                                         greater_is_better=True,
                                         needs_proba=True)
top_bottom_precision_scorer = make_scorer(top_bottom_precision_score,
                                          greater_is_better=True,
                                          needs_proba=True)
top_bottom_recall_scorer = make_scorer(top_bottom_recall_score,
                                       greater_is_better=True,
                                       needs_proba=True)
top_bottom_f1_scorer = make_scorer(top_bottom_f1_score,
                                   greater_is_better=True,
                                   needs_proba=True)

top_bottom_roc_auc_scorer = make_scorer(top_bottom_roc_auc_score,
                                        greater_is_better=True,
                                        needs_proba=True)

neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                                  needs_proba=True)

log_loss_scorer = make_scorer(log_loss,
                              greater_is_better=True,
                              needs_proba=True)

top_bottom_neg_log_loss_scorer = make_scorer(top_bottom_log_loss,
                                             greater_is_better=False,
                                             needs_proba=True)

top_bottom_log_loss_scorer = make_scorer(top_bottom_log_loss,
                                         greater_is_better=True,
                                         needs_proba=True)

# Standard regression scores
root_mean_squared_error_scorer = make_scorer(root_mean_squared_error,
                                             greater_is_better=False)

mean_absolute_percentage_error_scorer = make_scorer(
    mean_absolute_percentage_error,
    greater_is_better=False)


SCORERS = dict(accuracy=accuracy_scorer,
               precision=precision_scorer,
               recall=recall_scorer,
               f1=f1_scorer,
               roc_auc=roc_auc_scorer,
               top_bottom_accuracy=top_bottom_accuracy_scorer,
               top_bottom_precision=top_bottom_precision_scorer,
               top_bottom_recall=top_bottom_recall_scorer,
               top_bottom_f1=top_bottom_f1_scorer,
               top_bottom_roc_auc=top_bottom_roc_auc_scorer,
               neg_log_loss=neg_log_loss_scorer,
               log_loss=log_loss_scorer,
               top_bottom_neg_log_loss=top_bottom_neg_log_loss_scorer,
               top_bottom_log_loss=top_bottom_log_loss_scorer,
               root_mean_squared_error=root_mean_squared_error_scorer,
               mean_absolute_percentage_error=mean_absolute_percentage_error_scorer)   # noqa
