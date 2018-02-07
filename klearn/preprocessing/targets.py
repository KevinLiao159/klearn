"""
This module is only for target engineering
"""

import warnings
from sklearn.base import BaseEstimator, TransformerMixin


warnings.warn("This module was deprecated. All functions and classes "
              "are moved to pipedream.feature_engineering.filter",
              DeprecationWarning)


class TargetClassifier(BaseEstimator, TransformerMixin):
    """
    This class will be responsible for transforming target variable
    into binary signals according to a given decision boundary

    NOTE: for now, it only supports pandas series/dataframe.

    Parameters
    ----------
    is_quantile: bool. Default is False
        - If is_quantile=False, it is the abolute boundary for classifying \
            target. True if target > decision_boundary,
            False if target < decision_boundary
        - If is_quantile=True, it is the relative boundary to the same group \
            for classifying target. True if target > qth quantile of the group,
            False if target < qth quantile of the group

    decision_boundary: float, (-inf, inf) when is_quantile=False.
                       float, (0, 1) when is_quantile=True.
                       Default is 0

    level: str, indicates index name to groupby for transformation
    """
    def __init__(self, is_quantile=False, decision_boundary=0, level='date'):
        self.decision_boundary = decision_boundary
        self.is_quantile = is_quantile
        self.level = level

    def fit(self, y):
        pass

    def transform(self, y):
        if self.is_quantile:
            return y.groupby(level=self.level)\
                    .transform(lambda x: x > x.quantile(self.decision_boundary))                  # noqa

        else:  # not quantile
            return y.groupby(level=self.level)\
                    .transform(lambda x: x > self.decision_boundary)
