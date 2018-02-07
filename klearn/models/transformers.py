"""
transformers for data transforming in the sklearn pipeline
eg. features pruning, feature extraction, feature selection,
    decomposition, manifolds, and so on...
"""

# Authors: Kevin Liao <kevin.lwk.liao@gmail.com>

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from gravity_learn.preprocessing import IdentityScaler
from gravity_learn.utils import check_is_fitted
from gravity_learn.utils import force_array
from gravity_learn.logger import get_logger

logger = get_logger('models.transformers')


__all__ = ['sigmoil',
           'logit',
           'IdentityTransformer',
           'RFECVpca',
           'PCArfecv']


def sigmoil(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(x):
    return np.log(x) - np.log(1 - x)


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    This is an identity transformer. It is useful in the context of a
    FeatureUnion. It accepts no parameters and returns the array as is.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class RFECVpca(BaseEstimator, TransformerMixin):
    """Transforms features by recursive pruning and PCA

    This transfomer performs recursive feature elimination first,\
    reserve the best n features, and leave the rest to PCA.
    (Can be Deprecated by FeatureUnion ?)

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method that updates a
        `coef_` attribute that holds the fitted parameters. Important features
        must correspond to high absolute values in the `coef_` array.

        For instance, this is the case for most supervised learning
        algorithms such as Support Vector Classifiers and Generalized
        Linear Models from the `svm` and `linear_model` modules.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int, default 1
        Number of cores to run in parallel while fitting across folds.
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.

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

    n_components : int, float, None or string
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        if n_components == 'mle' and svd_solver == 'full', Minka\'s MLE is used
        to guess the dimension
        if ``0 < n_components < 1`` and svd_solver == 'full', select the number
        of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components
        n_components cannot be equal to n_features for svd_solver == 'arpack'.

    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < X.shape[1]
        randomized :
            run randomized SVD by the method of Halko et al.

        .. versionadded:: 0.18.0

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.

        .. versionadded:: 0.18.0

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

        .. versionadded:: 0.18.0

    random_state : int or RandomState instance or None (default None)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton. Used by svd_solver == 'arpack' or 'randomized'.

    Attributes
    ----------
    rfe_n_features_ : int
        The number of selected features by recursive feature elimination.

    rfe_support_ : array of shape [n_features]
        The mask of selected features by optimizaton

    rfe_ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    rfe_grid_scores_ : array of shape [n_subsets_of_features]
        The cross-validation scores such that
        ``grid_scores_[i]`` corresponds to
        the CV score of the i-th subset of features.

    rfe_estimator_ : object
        The external estimator fit on the reduced dataset.

        components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.

    pca_explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.

        .. versionadded:: 0.18

    pca_explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0.

    pca_mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=1)`.

    pca_n_components_ : int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or n_features if n_components is None.

    pca_noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        computed the estimated data covariance and score samples.

    References
    ----------
    For n_components == 'mle', this class uses the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`

    Implements the probabilistic PCA model from:
    M. Tipping and C. Bishop, Probabilistic Principal Component Analysis,
    Journal of the Royal Statistical Society, Series B, 61, Part 3, pp. 611-622
    via the score and score_samples methods.
    See http://www.miketipping.com/papers/met-mppca.pdf

    For svd_solver == 'arpack', refer to `scipy.sparse.linalg.svds`.

    For svd_solver == 'randomized', see:
    `Finding structure with randomness: Stochastic algorithms
    for constructing approximate matrix decompositions Halko, et al., 2009
    (arXiv:909)`
    `A randomized algorithm for the decomposition of matrices
    Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert`
    """
    def __init__(self, estimator, step=1, cv=None, scoring=None, verbose=0,
                 n_jobs=1, pre_dispatch='2*n_jobs',
                 n_components=None, copy=True, whiten=False, svd_solver='auto',
                 tol=1E-10, iterated_power='auto', random_state=None,
                 standardizer=None):
        # RFE params
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        # PCA params
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        # standardizer
        if standardizer is None:
            self.standardizer = IdentityScaler()
        else:
            if not hasattr(standardizer, 'fit_transform'):
                raise TypeError("standardizer should be an object or "
                                "transformers with fit and transform "
                                "implementation.")
            self.standardizer = standardizer

    def _rfecv_fit(self, X, y):
        """First step: perform rfecv

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]

        y: array-like, shape[n_samples,].
        """
        # TODO: this is a temporary hack for old version simulator
        if y is not None:
            y = LabelEncoder().fit_transform(y)

        rfecv = RFECV(estimator=self.estimator, step=self.step, cv=self.cv,
                      scoring=self.scoring, verbose=self.verbose,
                      n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
        rfecv.fit(X, y)
        if self.verbose > 1:
            logger.info(
                "Feature pruning is done. "
                "{} features are preserved".format(sum(rfecv.support_))
            )
        # set final attributes
        self.rfecv = rfecv
        return self

    def _pca_fit(self, X, y=None):
        """
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        """
        pca = PCA(n_components=self.n_components, copy=self.copy,
                  whiten=self.whiten, svd_solver=self.svd_solver,
                  tol=self.tol, iterated_power=self.iterated_power,
                  random_state=self.random_state)
        pca.fit(X)
        if self.verbose > 1:
            logger.info(
                "PCA is done. "
                "{} components are preserved".format(pca.n_components_)
            )
        # set final attributes
        self.pca = pca
        return self

    def _fit(self, X, y):
        """Pipeline: RFECV, standardizer, PCA

        Parameters
        ----------
        X: array-like, shape [n_samples, n_features]

        y: array, shape[n_samples,].
        """
        # first step: RFECV
        if self.verbose > 1:
            logger.info("First step is recursive feature pruning")
        self._rfecv_fit(X, y)
        self.cols_for_pca = np.logical_not(self.rfecv.support_)
        if sum(self.cols_for_pca) > 1:
            # second step: preprocess: standardizer
            if self.verbose > 1:
                logger.info("Second step is standardizing data")
            X_std = self.standardizer.fit_transform(
                X[:, self.cols_for_pca])
            # last step: PCA
            if self.verbose > 1:
                logger.info("Third step is performing PCA")
            # TODO: debug mle
            logger.info(X_std.shape)
            self._pca_fit(X=X_std)
            return self
        else:
            # special case: no feature is eliminated
            if self.verbose > 0:
                logger.info("All features are reserved, no PCA needed")
                return self

    def fit(self, X, y):
        """Pipeline: RFECV, standardizer, standardizer, PCA

        Parameters
        ----------
        X: array-like, shape [n_samples, n_features]

        y: array, shape[n_samples,].
        """
        if self.copy:
            X, y = X.copy(), y.copy()
        X, y = force_array(X), force_array(y)
        self._fit(X, y)
        return self

    def transform(self, X):
        """Merge reserved df with PCA

        Parameters
        ----------
        X: array-like, shape [n_samples, n_features]

        Returns
        -------
        X : new array with dimension reduction,
          shape [n_samples, n_features]
        """
        check_is_fitted(self, 'pca')
        if self.copy:
            X = X.copy()
        X = force_array(X)
        # implement RFECV transform method
        X_reserved = X[:, self.rfecv.support_]
        if sum(self.cols_for_pca) > 1:
            # converted
            X_pca = \
                self.pca.transform(
                    self.standardizer.fit_transform(X[:, self.cols_for_pca])
                    )
            return np.hstack((X_reserved, X_pca))
        else:
            # speical case: no feature is eliminated
            return X_reserved


class PCArfecv(RFECVpca):
    """Transforms features by PCA first then recursive pruning

    This transfomer performs PCA first, then performs \
    recursive feature elimination
    """

    def __init__(self, estimator, step=1, cv=None, scoring=None, verbose=0,
                 n_jobs=1, pre_dispatch='2*n_jobs',
                 n_components=None, copy=True, whiten=False, svd_solver='auto',
                 tol=1E-10, iterated_power='auto', random_state=None,
                 standardizer=None):
        super(PCArfecv, self).__init__(estimator=estimator,
                                       step=step,
                                       cv=cv, scoring=scoring,
                                       verbose=verbose,
                                       n_jobs=n_jobs,
                                       pre_dispatch=pre_dispatch,
                                       n_components=n_components,
                                       copy=copy,
                                       whiten=whiten,
                                       svd_solver=svd_solver, tol=tol,
                                       iterated_power=iterated_power,
                                       random_state=random_state,
                                       standardizer=standardizer)

    def _fit(self, X, y):
        """Pipeline: standardizer, PCA, RFECV

        Parameters
        ----------
        X: array-like, shape [n_samples, n_features]

        y: array, shape[n_samples,].
        """
        # first step: standardization
        if self.verbose > 1:
            logger.info("First step is Standardization")
            X_std = self.standardizer.fit_transform(X)
        # second step: PCA
        if self.verbose > 1:
            logger.info("Second step is performing PCA")
        self._pca_fit(X=X_std)

        # last step: feature pruning
        if self.pca.n_components_ > 1:
            if self.verbose > 1:
                logger.info("Last step is recursive feature pruning")
            X_pca = self.pca.transform(X_std)

            self._rfecv_fit(X_pca, y)
        else:
            if self.verbose > 1:
                logger.info("Only one compenent left. No pruning needed")
        return self

    def transform(self, X):
        """Return final transformed df_X

        Parameters
        ----------
        X: array-like, shape [n_samples, n_features]

        Returns
        -------
        X : new array with dimension reduction,
            shape [n_samples, n_features]
        """
        check_is_fitted(self, 'rfecv')
        if self.copy:
            X = X.copy()
        X = force_array(X)
        # first step - PCA transformation
        X_pca = self.pca.transform(self.standardizer.fit_transform(X))
        if self.pca.n_components_ > 1:
            X_pruned = self.rfecv.transform(X_pca)
            return X_pruned
        else:
            return X_pca
