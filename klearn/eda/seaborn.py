# TODO: this is for wrapper functions on ploting raw data (no data preprocess)
# 1. density plot (list of cols) distplot / kdeplot
# optional (normality plot check/ stats.probplot)
# 2. linear fit (list of x against one y)
# 3. joint density (list of x against one y) with marginal distribution
# optional (violin plot --- for list of features with flag/color code)
# optional (violin plot --- for time series of a feature with flag/color code)
# optional (sns.boxplot for catergorical data --- don't have use case yet)
# make sure all wrapper are generalizable as much as possible
# (take a list of input instead of dataframe)

# Authors: Kevin Liao

"""seaborn wrapper for multiple subplots"""
import numpy as np
import pandas as pd
import seaborn.apionly as sns
import matplotlib.pyplot as plt

from gravity_learn.logger import get_logger, Loggable
logger = get_logger('eda.seaborn')

plt.style.use('ggplot')


class SeabornPlot(Loggable):
    """
    Wrapper for Seaborn API.
    Use cases:
        1. it provides a convenient way to plot multiple subplots in one run
        2. it gives quick and dirty EDA on every feature verse the target

    NOTE:
        1. It only takes pandas dataframe for the time being
        2. It only support nrows

    Parameters
    ----------
    nrows, ncols : int, optional, default: 1
        Number of rows/columns of the subplot grid.

    size : tuple (length, height), it is size per subplot

    sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
        Controls sharing of properties among x (`sharex`) or y (`sharey`)
        axes:

            - True or 'all': x- or y-axis will be shared among all
              subplots.
            - False or 'none': each subplot x- or y-axis will be
              independent.
            - 'row': each subplot row will share an x- or y-axis.
            - 'col': each subplot column will share an x- or y-axis.

        When subplots have a shared x-axis along a column, only the x tick
        labels of the bottom subplot are visible.  Similarly, when subplots
        have a shared y-axis along a row, only the y tick labels of the first
        column subplot are visible.

    **fig_kw :
        All additional keyword arguments are passed to the :func:`figure` call.

    """
    def __init__(self, size=(12, 6), sharex=False,
                 title_fontsize=20, label_fontsize=15,
                 tick_fontsize=12, **fig_kw):
        self.size = size
        self.sharex = sharex
        self.title_fontsize = tick_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = label_fontsize
        self.fig_kw = fig_kw

    # --------------------------------------------------
    #  Univariate Distribution Plots
    # --------------------------------------------------

    def distplot(self, x=None, data=None, *args, **kwargs):
        """
        Flexibly plot a univariate distribution of observations

        Parameters
        ----------
        x : list of str, input variables; these should be column names in data

        data : pandas dataframe

        **kwargs : other arguments in seaborn.distplot

            bins : argument for matplotlib hist(), or None, optional

            hist : bool, optional whether to plot a (normed) histogram

            kde : bool, optional, whether to plot a gaussian kernel \
                density estimate

            rug : bool, optional whether to draw a rugplot on the support axis

            fit : random variable object, optional

            color : matplotlib color, optional

            vertical : bool, optional

            norm_hist : bool, optional

            axlabel : string, False, or None, optional

            label : string, optional

        Returns
        -------
        figure : matplotlib figure with multiple axes

        References
        ----------
        Seaborn distplot further documentation
        https://seaborn.pydata.org/generated/seaborn.distplot.html
        """
        # check data
        if not isinstance(data, (pd.DataFrame)):
            raise ValueError('data must be pandas dataframe')

        # handle single string
        if not isinstance(x, (list, tuple, np.ndarray, pd.Index)):
            x = [x]

        # create fig and axes
        nrows = len(x)
        plt.close()
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex=self.sharex,
            figsize=(self.size[0], nrows * self.size[1])
        )
        # HACK: handle Axes indexing when only one ax in fig
        if nrows == 1:
            axes = [axes]
        # iterate thru x
        for i, col in enumerate(x):
            # check if col in data
            if col not in data.columns.values:
                raise ValueError('{} is NOT in data'.format(col))
            a = data[col]
            if np.logical_not(np.isfinite(a)).any():
                logger.warning('RUNTIME WARNING: {} column has inf or nan '
                               ''.format(col))
                a = a.replace([-np.inf, np.inf], np.nan).dropna()
            sns.distplot(a=a, data=data, ax=axes[i], *args, **kwargs)
            axes[i].set_title(
                label='Univariate Distribution of {}'.format(col),
                fontsize=self.title_fontsize)
            axes[i].set_xlabel(xlabel=col, fontsize=self.label_fontsize)
            axes[i].set_ylabel(ylabel='frequency',
                               fontsize=self.label_fontsize)
            axes[i].tick_params(axis='both',
                                which='maj',
                                labelsize=self.tick_fontsize)
            fig.subplots_adjust(
                wspace=0.5,
                hspace=0.3,
                left=0.125,
                right=0.9,
                top=0.9,
                bottom=0.1
            )
            fig.tight_layout()
        plt.show()

    # --------------------------------------------------
    #  Joint Distribution Plots
    # --------------------------------------------------

    def kdeplot(self, x=None, y=None, data=None, *args, **kwargs):
        """
        Fit and plot a univariate or bivariate kernel density estimate

        Parameters
        ----------
        x : a list of names of variable in data that need to visualize \
            their distribution

        y : a list of names of variable in data that need to visualize \
            its joint distribution against every x above

        data : pandas dataframe

        **kwargs : other arguments in seaborn.kdeplot

            shade : bool, optional

            vertical : bool, optional

            kernel : {'gau' | 'cos' | 'biw' | 'epa' | 'tri' | 'triw' }

            bw : {'scott' | 'silverman' | scalar | pair of scalars }, optional

            gridsize : int, optional

            cut : scalar, optional

            clip : pair of scalars, or pair of pair of scalars, optional

            legend : bool, optional

            cumulative : bool, optional

            shade_lowest : bool, optional

            cbar : bool, optional

            cbar_ax : matplotlib axes, optional

            cbar_kws : dict, optional

        Returns
        -------
        figure : matplotlib figure with multiple axes

        References
        ----------
        Seaborn distplot further documentation
        https://seaborn.pydata.org/generated/seaborn.kdeplot.html
        """
        # check data
        if not isinstance(data, (pd.DataFrame)):
            raise ValueError('data must be pandas dataframe')

        # check x and y
        if x is None:
            raise ValueError('x can NOT be None')
        else:  # x is NOT None
            if not isinstance(x, (list, tuple, np.ndarray, pd.Index)):
                x = [x]

        if not isinstance(y, (list, tuple, np.ndarray, pd.Index)):
            y = [y]

        # create fig and axes
        nrows = len(y) * len(x)
        plt.close()
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex=self.sharex,
            figsize=(self.size[0], nrows * self.size[1])
        )
        # HACK: handle Axes indexing when only one ax in fig
        if nrows == 1:
            axes = [axes]
        # iterate thru x
        for i, col_y in enumerate(y):
            if col_y is not None:
                if col_y not in data.columns.values:
                    raise ValueError('{} is NOT in data'.format(col_y))
                b = data[col_y]
                b_not_nan = np.ones(b.shape[0], dtype=np.bool)
                if np.logical_not(np.isfinite(b)).any():
                    logger.warning('RUNTIME WARNING: {} column has inf or nan '
                                   ''.format(col_y))
                    b = b.replace([-np.inf, np.inf], np.nan)
                    # filter
                    b_not_nan = np.logical_not(b.isnull())
            elif col_y is None:
                b = None
                b_not_nan = np.ones(data.shape[0], dtype=np.bool)
            # get ax count
            axes_start_count = i * len(x)

            for j, col_x in enumerate(x):
                # get ax location
                ax_loc = axes_start_count + j
                # check if col in data
                if col_x not in data.columns.values:
                    raise ValueError('{} is NOT in data'.format(col_x))
                a = data[col_x]
                a_not_nan = np.ones(a.shape[0], dtype=np.bool)
                if np.logical_not(np.isfinite(a)).any():
                    logger.warning('RUNTIME WARNING: {} column has inf or '
                                   'nan'.format(col_x))
                    a = a.replace([-np.inf, np.inf], np.nan)
                    # filter
                    a_not_nan = np.logical_not(a.isnull())
                # joint filter
                not_nan = b_not_nan & a_not_nan
                a = a[not_nan]
                if b is not None:
                    b = b[not_nan]
                sns.kdeplot(data=a, data2=b, ax=axes[ax_loc],
                            *args, **kwargs)
                if b is not None:
                    axes[ax_loc].set_title(
                        label='Joint Distribution of {} and {} '
                              ''.format(col_y, col_x),
                        fontsize=self.title_fontsize)
                    axes[ax_loc].set_xlabel(
                        xlabel=col_x,
                        fontsize=self.label_fontsize)
                    axes[ax_loc].set_ylabel(
                        ylabel=col_y,
                        fontsize=self.label_fontsize)
                else:  # b is None
                    axes[ax_loc].set_title(
                        label='Distribution of {}'.format(col_x),
                        fontsize=self.title_fontsize)
                    axes[ax_loc].set_xlabel(
                        xlabel=col_x,
                        fontsize=self.label_fontsize)
                    axes[ax_loc].set_ylabel(
                        ylabel='frequency',
                        fontsize=self.label_fontsize)
                axes[ax_loc].tick_params(
                    axis='both',
                    which='maj',
                    labelsize=self.tick_fontsize)
                axes[ax_loc].legend(loc='lower right')
                fig.subplots_adjust(
                    wspace=0.5,
                    hspace=0.3,
                    left=0.125,
                    right=0.9,
                    top=0.9,
                    bottom=0.1
                )
                fig.tight_layout()
        plt.show()

    def jointplot(self, x=None, y=None, data=None, *args, **kwargs):
        """
        Fit and plot a univariate or bivariate kernel density estimate

        Parameters
        ----------
        x : a list of names of variable in data that need to visualize \
            their distribution

        y : a list of names of variable in data that need to visualize \
            its joint distribution against every x above

        data : pandas dataframe

        **kwargs : other arguments in seaborn.jointplot

            kind : { 'scatter' | 'reg' | 'resid' | 'kde' | 'hex' }, optional

            stat_func : callable or None, optional

            color : matplotlib color, optional

            size : numeric, optional

            ratio : numeric, optional

            space : numeric, optional

            dropna : bool, optional

            {x, y}lim : two-tuples, optional

            {joint, marginal, annot}_kws : dicts, optional

        Returns
        -------
        JointGrid object with the plot on it

        References
        ----------
        Seaborn distplot further documentation
        https://seaborn.pydata.org/generated/seaborn.jointplot.html
        """
        # check data
        if not isinstance(data, (pd.DataFrame)):
            raise ValueError('data must be pandas dataframe')

        # check x and y
        if x is None:
            raise ValueError('x can NOT be None')
        else:  # x is NOT None
            if not isinstance(x, (list, tuple, np.ndarray, pd.Index)):
                x = [x]
        if y is None:
            raise ValueError('y can NOT be None')
        else:  # y is NOT None
            if not isinstance(y, (list, tuple, np.ndarray, pd.Index)):
                y = [y]

        # no figure configuration needed
        plt.close()
        # iterate thru x
        for i, col_y in enumerate(y):
            if col_y not in data.columns.values:
                raise ValueError('{} is NOT in data'.format(col_y))
            b = data[col_y]
            b_not_nan = np.ones(b.shape[0], dtype=np.bool)
            if np.logical_not(np.isfinite(b)).any():
                logger.warning('RUNTIME WARNING: {} column has inf or nan '
                               ''.format(col_y))
                b = b.replace([-np.inf, np.inf], np.nan)
                # filter
                b_not_nan = np.logical_not(b.isnull())

            for j, col_x in enumerate(x):
                # check if col in data
                if col_x not in data.columns.values:
                    raise ValueError('{} is NOT in data'.format(col_x))
                a = data[col_x]
                a_not_nan = np.ones(a.shape[0], dtype=np.bool)
                if np.logical_not(np.isfinite(a)).any():
                    logger.warning('RUNTIME WARNING: {} column has inf or '
                                   'nan'.format(col_x))
                    a = a.replace([-np.inf, np.inf], np.nan)
                    # filter
                    a_not_nan = np.logical_not(a.isnull())
                # joint filter
                not_nan = b_not_nan & a_not_nan
                joint_grid = sns.jointplot(
                    x=a[not_nan],
                    y=b[not_nan],
                    size=self.size[0],
                    *args, **kwargs)

                joint_grid.fig.axes[1].set_title(
                        label='Joint Distribution of {} and {} '
                              ''.format(col_y, col_x),
                        fontsize=self.title_fontsize)
                joint_grid.fig.axes[0].set_xlabel(
                    xlabel=col_x,
                    fontsize=self.label_fontsize)
                joint_grid.fig.axes[0].set_ylabel(
                    ylabel=col_y,
                    fontsize=self.label_fontsize)
                joint_grid.fig.axes[0].tick_params(
                    axis='both',
                    which='maj',
                    labelsize=self.tick_fontsize)
                joint_grid.fig.axes[0].legend(loc='upper right')
                joint_grid.fig.subplots_adjust(
                    wspace=0.5,
                    hspace=0.3,
                    left=0.125,
                    right=0.9,
                    top=0.9,
                    bottom=0.1
                )
                joint_grid.fig.tight_layout()
        plt.show()

    def lmplot(self, x=None, y=None, hue=None, data=None, *args, **kwargs):
        """
        Plot data and regression model fits

        Parameters
        ----------
        x : a list of names of variable in data that need to visualize \
            their distribution

        y : a list of names of variable in data that need to visualize \
            its joint distribution against every x above

        hue : the name of a variable in data that provides labels for each \
            category

        data : pandas dataframe

        **kwargs : other arguments in seaborn.jointplot

            palette : palette name, list, or dict, optional

            col_wrap : int, optional

            size : scalar, optional

            aspect : scalar, optional

            markers : matplotlib marker code or list of marker codes, optional

            share{x,y} : bool, optional

            legend : bool, optional

            legend_out : bool, optional

            x_estimator : callable that maps vector -> scalar, optional

            x_bins : int or vector, optional

            x_ci : 'ci', 'sd', int in [0, 100] or None, optional

            scatter : bool, optional

            fit_reg : bool, optional

            ci : int in [0, 100] or None, optional

            n_boot : int, optional

            units : variable name in data, optional

            order : int, optional

            logistic : bool, optional

            lowess : bool, optional

            robust : bool, optional

            logx : bool, optional

            {x,y}_partial : strings in data or matrices

            truncate : bool, optional

            {x,y}_jitter : floats, optional

            {scatter,line}_kws : dictionaries

        Returns
        -------
        JointGrid object with the plot on it

        References
        ----------
        Seaborn distplot further documentation
        https://seaborn.pydata.org/generated/seaborn.seaborn.lmplot
        """
        # check data
        if not isinstance(data, (pd.DataFrame)):
            raise ValueError('data must be pandas dataframe')

        # check x and y
        if x is None:
            raise ValueError('x can NOT be None')
        else:  # x is NOT None
            if not isinstance(x, (list, tuple, np.ndarray, pd.Index)):
                x = [x]
        if y is None:
            raise ValueError('y can NOT be None')
        else:  # y is NOT None
            if not isinstance(y, (list, tuple, np.ndarray, pd.Index)):
                y = [y]
        if hue is not None:
            if hue not in data.columns.values:
                raise ValueError('{} is NOT in data'.format(hue))

        # no figure configuration needed
        plt.close()
        # iterate thru x
        for i, col_y in enumerate(y):
            if col_y not in data.columns.values:
                raise ValueError('{} is NOT in data'.format(col_y))
            b = data[col_y]
            b_not_nan = np.ones(b.shape[0], dtype=np.bool)
            if np.logical_not(np.isfinite(b)).any():
                logger.warning('RUNTIME WARNING: {} column has inf or nan '
                               ''.format(col_y))
                b = b.replace([-np.inf, np.inf], np.nan)
                # filter
                b_not_nan = np.logical_not(b.isnull())

            for j, col_x in enumerate(x):
                # check if col in data
                if col_x not in data.columns.values:
                    raise ValueError('{} is NOT in data'.format(col_x))
                a = data[col_x]
                a_not_nan = np.ones(a.shape[0], dtype=np.bool)
                if np.logical_not(np.isfinite(a)).any():
                    logger.warning('RUNTIME WARNING: {} column has inf or '
                                   'nan'.format(col_x))
                    a = a.replace([-np.inf, np.inf], np.nan)
                    # filter
                    a_not_nan = np.logical_not(a.isnull())
                # joint filter
                not_nan = b_not_nan & a_not_nan
                joint_grid = sns.lmplot(
                    x=col_x,
                    y=col_y,
                    data=data.loc[not_nan, :],
                    hue=hue,
                    legend=True,
                    legend_out=False,
                    size=self.size[0],
                    *args, **kwargs)

                joint_grid.fig.axes[0].set_title(
                        label='Reg Fit of {} on {} '
                              ''.format(col_y, col_x),
                        fontsize=self.title_fontsize)
                joint_grid.fig.axes[0].set_xlabel(
                    xlabel=col_x,
                    fontsize=self.label_fontsize)
                joint_grid.fig.axes[0].set_ylabel(
                    ylabel=col_y,
                    fontsize=self.label_fontsize)
                joint_grid.fig.axes[0].tick_params(
                    axis='both',
                    which='maj',
                    labelsize=self.tick_fontsize)
                joint_grid.fig.axes[0].legend(loc='upper right')
                joint_grid.fig.subplots_adjust(
                    wspace=0.5,
                    hspace=0.3,
                    left=0.125,
                    right=0.9,
                    top=0.9,
                    bottom=0.1
                )
                joint_grid.fig.tight_layout()
        plt.show()

    # --------------------------------------------------
    #  Categorical Plots
    # --------------------------------------------------

    def boxplot(self, x=None, y=None, hue=None, data=None,
                *args, **kwargs):
        """
        Draw a box plot to show distributions with respect to categories

        Parameters
        ----------
        x : the name of a variable in data that provides labels for categories

        y : a list of names of variables in data that need to visualize \
            distribution

        hue : the name of a variable in data that provides labels for \
            sub-categories in each big category

        data : pandas dataframe

        **kwargs : other arguments in seaborn.boxplot

            order, hue_order : lists of strings, optional

            orient : 'v' | 'h', optional

            color : matplotlib color, optional

            palette : palette name, list, or dict, optional

            saturation : float, optional

            width : float, optional

            dodge : bool, optional

            fliersize : float, optional

            linewidth : float, optional

            whis : float, optional

            notch : boolean, optional

        Returns
        -------
        figure : matplotlib figure with multiple axes

        References
        ----------
        Seaborn distplot further documentation
        https://seaborn.pydata.org/generated/seaborn.boxplot.html
        """
        # check data
        if not isinstance(data, (pd.DataFrame)):
            raise ValueError('data must be pandas dataframe')

        # check x and hue
        if x is not None:
            if x not in data.columns.values:
                raise ValueError('{} is NOT in data'.format(x))
        if hue is not None:
            if hue not in data.columns.values:
                raise ValueError('{} is NOT in data'.format(hue))

        # handle single string
        if not isinstance(y, (list, tuple, np.ndarray, pd.Index)):
            y = [y]

        # create fig and axes
        nrows = len(y)
        plt.close()
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex=self.sharex,
            figsize=(self.size[0], nrows * self.size[1])
        )
        # HACK: handle Axes indexing when only one ax in fig
        if nrows == 1:
            axes = [axes]
        # iterate thru x
        for i, col in enumerate(y):
            # check if col in data
            if col not in data.columns.values:
                raise ValueError('{} is NOT in data'.format(col))
            a = data[col]
            if np.logical_not(np.isfinite(a)).any():
                logger.warning('RUNTIME WARNING: {} column has inf or nan '
                               ''.format(col))
                a = a.replace([-np.inf, np.inf], np.nan)
                # filter
                not_nan = np.logical_not(a.isnull())
                a = a[not_nan]
                if x is not None:
                    x = data[x][not_nan]
                if hue is not None:
                    hue = data[hue][not_nan]

            sns.boxplot(x=x, y=a, hue=hue, data=data, ax=axes[i],
                        *args, **kwargs)
            if x is not None:
                axes[i].set_title(
                    label='Box Distribution of {} With Respect To {} '
                          ''.format(col, x),
                    fontsize=self.title_fontsize)
                axes[i].set_xlabel(xlabel=x, fontsize=self.label_fontsize)
                axes[i].set_ylabel(ylabel=col,
                                   fontsize=self.label_fontsize)
            else:  # x is None
                axes[i].set_title(
                    label='Box Distribution of {}'.format(col),
                    fontsize=self.title_fontsize)
                axes[i].set_xlabel(xlabel=col, fontsize=self.label_fontsize)
                axes[i].set_ylabel(ylabel='value',
                                   fontsize=self.label_fontsize)
            axes[i].tick_params(axis='both',
                                which='maj',
                                labelsize=self.tick_fontsize)
            axes[i].legend(loc='lower right')
            fig.subplots_adjust(
                wspace=0.5,
                hspace=0.3,
                left=0.125,
                right=0.9,
                top=0.9,
                bottom=0.1
            )
            fig.tight_layout()
        plt.show()

    def violinplot(self, x=None, y=None, hue=None, data=None,
                   *args, **kwargs):
        """
        Draw a box plot to show distributions with respect to categories

        Parameters
        ----------
        x : the name of a variable in data that provides labels for categories

        y : a list of names of variables in data that needs to visualize \
            distribution

        hue : the name of a variable in data that provides labels for \
            sub-categories in each big category

        data : pandas dataframe

        **kwargs : other arguments in seaborn.distplot

            order, hue_order : lists of strings, optional

            bw : {'scott', 'silverman', float}, optional

            cut : float, optional

            scale : {'area', 'count', 'width'}, optional

            scale_hue : bool, optional

            gridsize : int, optional

            width : float, optional

            inner : {'box', 'quartile', 'point', 'stick', None}, optional

            split : bool, optional

            dodge : bool, optional

            orient : 'v' | 'h', optional

            linewidth : float, optional

            color : matplotlib color, optional

            palette : palette name, list, or dict, optional

            saturation : float, optional

        Returns
        -------
        figure : matplotlib figure with multiple axes

        References
        ----------
        Seaborn distplot further documentation
        https://seaborn.pydata.org/generated/seaborn.violinplot.html
        """
        # check data
        if not isinstance(data, (pd.DataFrame)):
            raise ValueError('data must be pandas dataframe')

        # check x and hue
        if x is not None:
            if x not in data.columns.values:
                raise ValueError('{} is NOT in data'.format(x))
        if hue is not None:
            if hue not in data.columns.values:
                raise ValueError('{} is NOT in data'.format(hue))

        # handle single string
        if not isinstance(y, (list, tuple, np.ndarray, pd.Index)):
            y = [y]

        # create fig and axes
        nrows = len(y)
        plt.close()
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex=self.sharex,
            figsize=(self.size[0], nrows * self.size[1])
        )
        # HACK: handle Axes indexing when only one ax in fig
        if nrows == 1:
            axes = [axes]
        # iterate thru x
        for i, col in enumerate(y):
            # check if col in data
            if col not in data.columns.values:
                raise ValueError('{} is NOT in data'.format(col))
            a = data[col]
            if np.logical_not(np.isfinite(a)).any():
                logger.warning('RUNTIME WARNING: {} column has inf or nan '
                               ''.format(col))
                a = a.replace([-np.inf, np.inf], np.nan)
                # filter
                not_nan = np.logical_not(a.isnull())
                a = a[not_nan]
                if x is not None:
                    x = data[x][not_nan]
                if hue is not None:
                    hue = data[hue][not_nan]

            sns.violinplot(x=x, y=a, hue=hue, data=data, ax=axes[i],
                           *args, **kwargs)
            if x is not None:
                axes[i].set_title(
                    label='Violin Distribution of {} With Respect To {} '
                          ''.format(col, x),
                    fontsize=self.title_fontsize)
                axes[i].set_xlabel(xlabel=x, fontsize=self.label_fontsize)
                axes[i].set_ylabel(ylabel=col,
                                   fontsize=self.label_fontsize)
            else:  # x is None
                axes[i].set_title(
                    label='Violin Distribution of {}'.format(col),
                    fontsize=self.title_fontsize)
                axes[i].set_xlabel(xlabel=col, fontsize=self.label_fontsize)
                axes[i].set_ylabel(ylabel='value',
                                   fontsize=self.label_fontsize)
            axes[i].tick_params(axis='both',
                                which='maj',
                                labelsize=self.tick_fontsize)
            axes[i].legend(loc='lower right')
            fig.subplots_adjust(
                wspace=0.5,
                hspace=0.3,
                left=0.125,
                right=0.9,
                top=0.9,
                bottom=0.1
            )
            fig.tight_layout()
        plt.show()

    # --------------------------------------------------
    #  Regression Plots
    # --------------------------------------------------
