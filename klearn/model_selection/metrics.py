"""
Metrics to assess performance on classification task given class prediction

Functions named as *_score return a scalar value to maximize: the higher
the better

Function named as *_error or *_loss return a scalar value to minimize:
the lower the better
"""

# Authors: Kevin Liao

import numpy as np
import math
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, log_loss)
from sklearn.metrics import mean_squared_error
from gravity_learn.utils import force_array, check_consistent_length


__all__ = ('top_bottom_accuracy_score',
           'top_bottom_precision_score',
           'top_bottom_recall_score',
           'top_bottom_f1_score',
           'top_bottom_roc_auc_score',
           'top_bottom_log_loss',
           'root_mean_squared_error',
           'mean_absolute_percentage_error')

# --------------------------------------------------
#  Classification metrics
# --------------------------------------------------


def _select_top_and_bottom(y_true, y_score,
                           top=50, bottom=50, interpolation='midpoint'):
    """
    Select truth values, predictions, scores of the top and bottom observations

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, ]
        True binary labels in binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, 2]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.

    top, bottom : float, int, or None, default 50.
        If int, it filters top/bottom n samples
        If float, it should be between 0.0 and 0.5 and it filters top/bottom x
        percentage of the entire data

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
    # check input
    check_consistent_length(y_true, y_score)
    y_true = force_array(y_true)
    y_score = force_array(y_score)
    n_samples = len(y_true)
    # convert float to int for top and bottom
    if isinstance(top, float):
        if not (0 < top < 0.5):
            raise ValueError('Warning! top is out of the range (0, 0.5)')
        top = int(round(top * n_samples))
    if isinstance(bottom, float):
        if not (0 < bottom < 0.5):
            raise ValueError('Warning! bottom is out of the range (0, 0.5)')
        bottom = int(round(bottom * n_samples))
    # get P1 (label one)
    p_one = y_score[:, 1]
    # filter top and bottom
    top_idx = np.argsort(p_one)[::-1][:top]
    bottom_idx = np.argsort(p_one)[:bottom]
    filter_idx = np.sort(np.concatenate([top_idx, bottom_idx]))
    # filtering
    y_true_ext = y_true[filter_idx]
    y_score_ext = y_score[filter_idx]
    y_pred_ext = y_score_ext[:, 1] >= 0.5
    return y_true_ext, y_score_ext, y_pred_ext


def top_bottom_accuracy_score(y_true, y_score, top=50, bottom=50,
                              interpolation='midpoint',
                              normalize=True, sample_weight=None):
    """
    Accuracy score of the top and bottom observations

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    top, bottom : float, int, or None, default 50.
        If int, it filters top/bottom n samples
        If float, it should be between 0.0 and 0.5 and it filters top/bottom x
        percentage of the entire data

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
        _select_top_and_bottom(
            y_true=y_true,
            y_score=y_score,
            top=top,
            bottom=bottom,
            interpolation=interpolation
        )
    return accuracy_score(y_true=y_true_ext, y_pred=y_pred_ext,
                          normalize=normalize, sample_weight=sample_weight)


def top_bottom_precision_score(y_true, y_score, top=50, bottom=50,
                               interpolation='midpoint',
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

    top, bottom : float, int, or None, default 50.
        If int, it filters top/bottom n samples
        If float, it should be between 0.0 and 0.5 and it filters top/bottom x
        percentage of the entire data

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
        _select_top_and_bottom(
            y_true=y_true,
            y_score=y_score,
            top=top,
            bottom=bottom,
            interpolation=interpolation
        )
    return precision_score(y_true=y_true_ext, y_pred=y_pred_ext,
                           pos_label=1, average=average,
                           sample_weight=sample_weight)


def top_bottom_recall_score(y_true, y_score, top=50, bottom=50,
                            interpolation='midpoint',
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

    top, bottom : float, int, or None, default 50.
        If int, it filters top/bottom n samples
        If float, it should be between 0.0 and 0.5 and it filters top/bottom x
        percentage of the entire data

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
        _select_top_and_bottom(
            y_true=y_true,
            y_score=y_score,
            top=top,
            bottom=bottom,
            interpolation=interpolation
        )
    return recall_score(y_true=y_true_ext, y_pred=y_pred_ext,
                        pos_label=1, average=average,
                        sample_weight=sample_weight)


def top_bottom_f1_score(y_true, y_score, top=50, bottom=50,
                        interpolation='midpoint',
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

    top, bottom : float, int, or None, default 50.
        If int, it filters top/bottom n samples
        If float, it should be between 0.0 and 0.5 and it filters top/bottom x
        percentage of the entire data

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
        _select_top_and_bottom(
            y_true=y_true,
            y_score=y_score,
            top=top,
            bottom=bottom,
            interpolation=interpolation
        )
    return f1_score(y_true=y_true_ext, y_pred=y_pred_ext,
                    pos_label=1, average=average,
                    sample_weight=sample_weight)


def top_bottom_roc_auc_score(y_true, y_score, top=50, bottom=50,
                             interpolation='midpoint',
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

    top, bottom : float, int, or None, default 50.
        If int, it filters top/bottom n samples
        If float, it should be between 0.0 and 0.5 and it filters top/bottom x
        percentage of the entire data

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
        _select_top_and_bottom(
            y_true=y_true,
            y_score=y_score,
            top=top,
            bottom=bottom,
            interpolation=interpolation
        )
    # Need a hack here [:,1]
    # TODO: need to handle pos_label in _binary_check
    # TODO: need to revisit
    return roc_auc_score(y_true=y_true_ext, y_score=y_score_ext[:, 1],
                         average=average, sample_weight=sample_weight)


def top_bottom_log_loss(y_true, y_score, top=50, bottom=50,
                        interpolation='midpoint',
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

    top, bottom : float, int, or None, default 50.
        If int, it filters top/bottom n samples
        If float, it should be between 0.0 and 0.5 and it filters top/bottom x
        percentage of the entire data

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
        _select_top_and_bottom(
            y_true=y_true,
            y_score=y_score,
            top=top,
            bottom=bottom,
            interpolation=interpolation
        )
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
