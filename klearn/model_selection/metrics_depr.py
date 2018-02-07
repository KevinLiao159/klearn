"""
Metrics to assess performance on classification task given class prediction

Functions named as *_score return a scalar value to maximize: the higher
the better

Function named as *_error or *_loss return a scalar value to minimize:
the lower the better
"""

# Authors: Kevin Liao <kevin.lwk.liao@gmail.com>

import numpy as np
import math
from sklearn.metrics.classification import _weighted_sum
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, log_loss)
from sklearn.metrics import mean_squared_error
from gravity_learn.utils import force_array

import warnings

warnings.warn("This module was deprecated. All scores and metrics "
              "are moved to model_selection.metrics",
              DeprecationWarning)

__all__ = ('classification_error',
           'long_error',
           'short_error',
           'short_precision_score',
           'short_recall_score',
           'top_bottom_accuracy_score',
           'top_bottom_error',
           'top_bottom_long_error',
           'top_bottom_short_error',
           'top_bottom_precision_score',
           'top_bottom_recall_score',
           'top_bottom_short_precision_score',
           'top_bottom_short_recall_score',
           'top_bottom_f1_score',
           'top_bottom_roc_auc_score',
           'top_bottom_log_loss',
           'root_mean_squared_error',
           'mean_absolute_percentage_error')

# --------------------------------------------------
#  Classification metrics
# --------------------------------------------------


def _select_top_and_bottom(y_true, y_score,
                           percentile=10, interpolation='midpoint'):
    """
    Select truth values, predictions, scores of the top and bottom observations

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, ]
        True binary labels in binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, 2]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.

    Returns
    -------
    y_true_ext : array, shape = [n_samples] or [n_samples, ]
        True binary labels in binary label indicators of top and bottom

    y_score_ext : array, shape = [n_samples] or [n_samples, 2]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions of top and bottom.

    y_pred_ext : array, shape = [n_samples] or [n_samples, ]
        Target prediction, can either be 1 or 0, top is always 1 and bottom\
        is always 0.
    """
    y_true = force_array(y_true)
    y_score = force_array(y_score)
    upperQ = np.percentile(y_score[:, 1], q=(100-percentile),
                           interpolation=interpolation)
    lowerQ = np.percentile(y_score[:, 1], q=percentile,
                           interpolation=interpolation)
    top_bottom_filter = (y_score[:, 1] >= upperQ) | (y_score[:, 1] <= lowerQ)

    y_true_ext = y_true[top_bottom_filter]
    y_score_ext = y_score[top_bottom_filter]
    y_pred_ext = y_score_ext[:, 1] >= 0.5
    return y_true_ext, y_score_ext, y_pred_ext


def classification_error(y_true, y_pred,
                         normalize=True, sample_weight=None):
    """
    Compute classification error

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    normalize : bool, optional (default=True)
        If False, return the number of misclassified samples.
        Otherwise, return the fraction of misclassified samples.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    error : float
        If normalize == True, return the misclassified samples
        (float), else it returns the number of misclassified samples
        (int).
    """
    return 1 - accuracy_score(y_true, y_pred, normalize, sample_weight)


def long_error(y_true, y_pred, normalize=True, sample_weight=None):
    """
    Error of long classification. False negative rate (FNR)

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
        If False, return the number of misclassified samples.
        Otherwise, return the fraction of misclassified samples.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    error : float
        If normalize == True, return the misclassified samples
        (float), else it returns the number of misclassified samples
        (int).

        The best performance is 0
    """
    long_true = y_true[y_true == 1]
    long_pred = y_pred[y_true == 1]
    score = long_pred != long_true
    return _weighted_sum(score, sample_weight, normalize)


def short_error(y_true, y_pred, normalize=True, sample_weight=None):
    """
    Error of short classification. False positive rate (FPR)

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
        If False, return the number of misclassified samples.
        Otherwise, return the fraction of misclassified samples.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    error : float
        If normalize == True, return the misclassified samples
        (float), else it returns the number of misclassified samples
        (int).

        The best performance is 0
    """
    short_true = y_true[y_true == 0]
    short_pred = y_pred[y_true == 0]
    score = short_pred != short_true

    return _weighted_sum(score, sample_weight, normalize)


def short_precision_score(y_true, y_pred,
                          average='binary', sample_weight=None):
    """
    Precision of short prediction. False omission rate (FOR)

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        'binary':
            Only report results for the class specified by pos_label.
            This is applicable only if targets (y_{true,pred}) are binary.
        'micro':
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        'macro':
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        'weighted':
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        'samples':
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Precision of the negative class in binary classification or weighted
        average of the precision of each class for the multiclass task.
    """
    p = precision_score(y_true, y_pred,
                        labels=None, pos_label=0,
                        average=average, sample_weight=sample_weight)
    return p


def short_recall_score(y_true, y_pred, average='binary', sample_weight=None):
    """
    Recall of short prediction. True negative rate (TNR), Specificity (SPC)

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        'binary':
            Only report results for the class specified by pos_label.
            This is applicable only if targets (y_{true,pred}) are binary.
        'micro':
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        'macro':
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        'weighted':
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        'samples':
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    recall: float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Recall of the negative class in binary classification or weighted
        average of the recall of each class for the multiclass task.
    """
    r = recall_score(y_true, y_pred,
                     labels=None, pos_label=0, average=average,
                     sample_weight=sample_weight)
    return r


def top_bottom_accuracy_score(y_true, y_score,
                              percentile=10, interpolation='midpoint',
                              normalize=True, sample_weight=None):
    """
    Accuracy score of top and bottom percentile observations.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.

    normalize : bool, optional (default=True)
        If False, return the number of misclassified samples.
        Otherwise, return the fraction of misclassified samples.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    score : float
        If normalize == True, return the misclassified samples
        (float), else it returns the number of misclassified samples
        (int).

        The best performance is 1 with normalize == True and the number
        of samples with normalize == False.
    """
    y_true_ext, y_score_ext, y_pred_ext =\
        _select_top_and_bottom(y_true, y_score, percentile, interpolation)

    return accuracy_score(y_true_ext, y_pred_ext, normalize, sample_weight)


def top_bottom_error(y_true, y_score,
                     percentile=10, interpolation='midpoint',
                     normalize=True, sample_weight=None):
    """
    Classification error for top and bottom percentile
    observations.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.

    normalize : bool, optional (default=True)
        If False, return the number of misclassified samples.
        Otherwise, return the fraction of misclassified samples.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    error : float
        If normalize == True, return the misclassified samples
        (float), else it returns the number of misclassified samples
        (int).

        The best performance is 0
    """
    y_true_ext, y_score_ext, y_pred_ext =\
        _select_top_and_bottom(y_true, y_score, percentile, interpolation)

    return classification_error(y_true_ext, y_pred_ext,
                                normalize, sample_weight)


def top_bottom_long_error(y_true, y_score,
                          percentile=10, interpolation='midpoint',
                          normalize=True, sample_weight=None):
    """
    Classification error for long class of top and bottom percentile
    observations.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.

    normalize : bool, optional (default=True)
        If False, return the number of misclassified samples.
        Otherwise, return the fraction of misclassified samples.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    error : float
        If normalize == True, return the misclassified samples
        (float), else it returns the number of misclassified samples
        (int).

        The best performance is 0
    """
    y_true_ext, y_score_ext, y_pred_ext =\
        _select_top_and_bottom(y_true, y_score, percentile, interpolation)

    return long_error(y_true_ext, y_pred_ext, normalize, sample_weight)


def top_bottom_short_error(y_true, y_score,
                           percentile=10, interpolation='midpoint',
                           normalize=True, sample_weight=None):
    """
    Classification error for short class of top and bottom percentile
    observations.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.

    normalize : bool, optional (default=True)
        If False, return the number of misclassified samples.
        Otherwise, return the fraction of misclassified samples.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    error : float
        If normalize == True, return the misclassified samples
        (float), else it returns the number of misclassified samples
        (int).

        The best performance is 0
    """
    y_true_ext, y_score_ext, y_pred_ext =\
        _select_top_and_bottom(y_true, y_score, percentile, interpolation)

    return short_error(y_true_ext, y_pred_ext, normalize, sample_weight)


def top_bottom_precision_score(y_true, y_score,
                               percentile=10, interpolation='midpoint',
                               average='binary', sample_weight=None):
    """
    Compute the precision of top and bottom observations

    The precision is the ratio tp / (tp + fp) where tp is the number of
    true positives and fp the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        'binary':
            Only report results for the class specified by pos_label.
            This is applicable only if targets (y_{true,pred}) are binary.
        'micro':
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        'macro':
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        'weighted':
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        'samples':
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Top and bottom precision of the positive class in binary \
        classification or weighted average of the precision of each class \
        for the multiclass task.
    """
    y_true_ext, y_score_ext, y_pred_ext =\
        _select_top_and_bottom(y_true, y_score, percentile, interpolation)

    return precision_score(y_true=y_true_ext, y_pred=y_pred_ext,
                           pos_label=1, average=average,
                           sample_weight=sample_weight)


def top_bottom_recall_score(y_true, y_score,
                            percentile=10, interpolation='midpoint',
                            average='binary', sample_weight=None):
    """
    Compute the recall of top and bottom observations

    The precision is the ratio tp / (tp + fp) where tp is the number of
    true positives and fp the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        'binary':
            Only report results for the class specified by pos_label.
            This is applicable only if targets (y_{true,pred}) are binary.
        'micro':
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        'macro':
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        'weighted':
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        'samples':
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Top and bottom recall of the positive class in binary classification \
        or weighted average of the precision of each class for \
        the multiclass task.
    """
    y_true_ext, y_score_ext, y_pred_ext =\
        _select_top_and_bottom(y_true, y_score, percentile, interpolation)

    return recall_score(y_true=y_true_ext, y_pred=y_pred_ext,
                        pos_label=1, average=average,
                        sample_weight=sample_weight)


def top_bottom_short_precision_score(y_true, y_score,
                                     percentile=10, interpolation='midpoint',
                                     average='binary', sample_weight=None):
    """
    Compute the short precision of top and bottom observations

    The precision is the ratio tp / (tp + fp) where tp is the number of
    true positives and fp the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        'binary':
            Only report results for the class specified by pos_label.
            This is applicable only if targets (y_{true,pred}) are binary.
        'micro':
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        'macro':
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        'weighted':
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        'samples':
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Top and bottom precision of the negative class in binary \
        classification or weighted average of the precision of each class \
        for the multiclass task.
    """
    y_true_ext, y_score_ext, y_pred_ext =\
        _select_top_and_bottom(y_true, y_score, percentile, interpolation)

    return precision_score(y_true=y_true_ext, y_pred=y_pred_ext,
                           pos_label=0, average=average,
                           sample_weight=sample_weight)


def top_bottom_short_recall_score(y_true, y_score,
                                  percentile=10, interpolation='midpoint',
                                  average='binary', sample_weight=None):
    """
    Compute the recall of top and bottom of short observations

    The precision is the ratio tp / (tp + fp) where tp is the number of
    true positives and fp the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        'binary':
            Only report results for the class specified by pos_label.
            This is applicable only if targets (y_{true,pred}) are binary.
        'micro':
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        'macro':
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        'weighted':
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        'samples':
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Top and bottom recall of the positive class in binary classification \
        or weighted average of the precision of each class for \
        the multiclass task.
    """
    y_true_ext, y_score_ext, y_pred_ext =\
        _select_top_and_bottom(y_true, y_score, percentile, interpolation)

    return recall_score(y_true=y_true_ext, y_pred=y_pred_ext,
                        pos_label=0, average=average,
                        sample_weight=sample_weight)


def top_bottom_f1_score(y_true, y_score,
                        percentile=10, interpolation='midpoint',
                        average='binary', sample_weight=None):
    """
    Compute the F1 score, also known as balanced F-score or F-measure

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    In the multi-class and multi-label case, this is the weighted average of
    the F1 score of each class.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        'binary':
            Only report results for the class specified by pos_label.
            This is applicable only if targets (y_{true,pred}) are binary.
        'micro':
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        'macro':
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        'weighted':
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        'samples':
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    f1_score : float or array of float, shape = [n_unique_labels]
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task.
    """
    y_true_ext, y_score_ext, y_pred_ext =\
        _select_top_and_bottom(y_true, y_score, percentile, interpolation)

    return f1_score(y_true=y_true_ext, y_pred=y_pred_ext,
                    pos_label=1, average=average,
                    sample_weight=sample_weight)


def top_bottom_roc_auc_score(y_true, y_score,
                             percentile=10, interpolation='midpoint',
                             average='macro', sample_weight=None):
    """
    Compute Area Under the Curve (AUC) from prediction scores
    Note: this implementation is restricted to the binary classification task
    or multilabel classification task in label indicator format.
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.
    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        'binary':
            Only report results for the class specified by pos_label.
            This is applicable only if targets (y_{true,pred}) are binary.
        'micro':
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        'macro':
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        'weighted':
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        'samples':
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    f1_score : float or array of float, shape = [n_unique_labels]
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task.
    """
    y_true_ext, y_score_ext, y_pred_ext =\
        _select_top_and_bottom(y_true, y_score, percentile, interpolation)
    # Need a hack here [:,1]
    # TODO: need to handle pos_label in _binary_check
    return roc_auc_score(y_true=y_true_ext, y_score=y_score_ext[:, 1],
                         average=average, sample_weight=sample_weight)


def top_bottom_log_loss(y_true, y_score,
                        percentile=10, interpolation='midpoint',
                        eps=1e-15, normalize=True, sample_weight=None):
    """Log loss, aka logistic loss or cross-entropy loss.

    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of the true labels given a probabilistic classifier's
    predictions. The log loss is only defined for two or more labels.
    For a single sample with true label yt in {0,1} and
    estimated probability yp that yt = 1, the log loss is

        -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    percentile: float, default 10 (10% quantile) 0 <= percentile <= 100,
        the top and bottom quantile(s) to select from all true values

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        New in version 0.18.0.
        This optional parameter specifies the interpolation method to use,\
        when the desired quantile lies between two data points i and j:
        linear: i + (j - i) * fraction, where fraction is the fractional part\
        of the index surrounded by i and j.
        lower: i.
        higher: j.
        nearest: i or j whichever is nearest.
        midpoint: (i + j) / 2.

    eps : float
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to max(eps, min(1 - eps, p)).

    normalize : bool, optional (default=True)
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    loss : float

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    y_true_ext, y_score_ext, y_pred_ext =\
        _select_top_and_bottom(y_true, y_score, percentile, interpolation)

    return log_loss(y_true=y_true_ext, y_pred=y_pred_ext,
                    eps=eps, normalize=normalize, sample_weight=sample_weight)


# --------------------------------------------------
#  Regression metrics
# --------------------------------------------------

def root_mean_squared_error(y_true, y_pred,
                            sample_weight=None,
                            multioutput='uniform_average'):

    """Root mean squared error regression loss

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape = (n_samples), optional
        Sample weights.

    multioutput : string in ['raw_values', 'uniform_average']
        or array-like of shape (n_outputs)
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    """
    mse = mean_squared_error(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput
    )
    return math.sqrt(mse)


def mean_absolute_percentage_error(y_true, y_pred, robust=False):
    """mean_absolute_percentage_error

    Use case:
        y is expressed in percent and we want to take pct into account

    Formula:
        mean_absolute_percentage_error = \
            mean(abs((y_true - y_pred) / y_true) * 100

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    robust : bool, if True, use median, otherwise, mean
        Default is False

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0)
    """
    y_true = force_array(y_true)
    y_pred = force_array(y_pred)
    if robust:
        loss = np.median(np.abs((y_true - y_pred)/y_true)) * 100
    else:  # use mean
        loss = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    return loss
