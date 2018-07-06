# Authors: Kevin Liao

"""Scatter plot (line plot) plotly wrapper"""
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
# from plotly import tools

from gravity_learn.utils import force_array, check_consistent_length
from gravity_learn.logger import get_logger

plotly.offline.init_notebook_mode(connected=True)

logger = get_logger(__name__)


__all__ = ['PlotlyPlot']


class PlotlyPlot:
    """
    Wrapper for plotly API, potentially generate line charts, scatter plots,
    bar charts, pie charts, box plots, stack charts, distribution plots

    Parameters
    ----------
    title: str, name of the plot title

    yaxis: dict, attributes of yaxis, eg. {'title': 'y', 'font': {'size': 12}}

    xaxis: dict, attributes of xaxis, eg. {'title': 'x', 'font': {'size': 12}}

    legend: dict, attributes of legend, eg. {'x': 0, 'y': 1, 'font': {}}

    width: int, default 900

    height: int, default 300
    """
    def __init__(self, title=None, yaxis=None, xaxis=None, legend=None,
                 width=900, height=500):
        self.title = title
        self.yaxis = yaxis
        self.xaxis = xaxis
        self.legend = legend
        self.width = width
        self.height = height
        if self.yaxis is None:
            self.yaxis = {}
        if self.xaxis is None:
            self.xaxis = {}
        if self.legend is None:
            self.legend = {}

    def lineplot(self, x, y,
                 legend_name=None, mode=None,
                 line_attribute=None,
                 marker_attribute=None,
                 **kwargs):
        """plot line(s)

        Parameters
        ----------
        y: a vector, or a list of vectors

        x: a vector

        legend_name: a name, or a list of names,
            must have same lenth as y

        mode: a mode, or a list of modes,
            must have same lenth as y
            eg. 'lines+markers' or ['lines', 'markers']

        line_attribute: a dict, or a list of dictionaries,
            must have same lenth as y
            eg. {'color': 'blue', 'dash': 'dot', 'width': 0.5}

        marker_attribute: a dict, or a list of dictionaries,
            must have same lenth as y
            eg. {
                    'size': 10,
                    'color': 'rgba(255, 182, 193, .9)',
                    'line': dict(width = 2)
                }

        **kwargs: a dictionary of additional params is passed in go.Scatter

        Returns
        -------
        A renderable plot
        """
        # make sure y is a list of vec
        if not isinstance(y, list):
            y = [y]
        # check if multiple vectors in y
        allowed_vector_type = \
            (list, pd.Series, np.ndarray, pd.DataFrame, pd.Index)
        if any(isinstance(vec, allowed_vector_type) for vec in y):
            # name
            if legend_name is not None:
                if not isinstance(legend_name, allowed_vector_type):
                    legend_name = [legend_name] * len(y)
                if len(legend_name) != len(y):
                    raise ValueError('length of legend_name must be the same as y') # noqa
            elif legend_name is None:
                legend_name = \
                    list(np.linspace(0, len(y)-1, len(y)).astype(int))
            # mode
            if mode is not None:
                if not isinstance(mode, allowed_vector_type):
                    mode = [mode] * len(y)
                if len(mode) != len(y):
                    raise ValueError('length of mode must be the same as y') # noqa
            elif mode is None:
                mode = ['lines'] * len(y)
            # line
            if line_attribute is not None:
                if not isinstance(line_attribute, allowed_vector_type):
                    line_attribute = [line_attribute] * len(y)
                if len(line_attribute) != len(y):
                    raise ValueError('length of line_attribute must be the same as y') # noqa
            elif line_attribute is None:
                line_attribute = [{}] * len(y)
            # marker
            if marker_attribute is not None:
                if not isinstance(marker_attribute, allowed_vector_type):
                    marker_attribute = [marker_attribute] * len(y)
                if len(marker_attribute) != len(y):
                    raise ValueError('length of marker_attribute must be the same as y') # noqa
            elif marker_attribute is None:
                marker_attribute = [{}] * len(y)

            # store data
            data = []
            for i, vec in enumerate(y):
                data.append(
                    go.Scatter(
                        x=x,
                        y=vec,
                        name=legend_name[i],
                        mode=mode[i],
                        line=line_attribute[i],
                        marker=marker_attribute[i],
                        **kwargs
                    )
                )
            layout = go.Layout(
                title=self.title,
                yaxis=self.yaxis,
                xaxis=self.xaxis,
                width=self.width,
                height=self.height,
                margin=go.Margin(
                    # l=50,
                    # r=50,
                    t=65,
                    b=60,
                    pad=4
                )
            )
            fig = go.Figure(data=data, layout=layout)
            plotly.offline.iplot(fig)

    def lineplot_y2(self, x, y, y2, y2_title=None,
                    legend_name=None, line_attribute=None,
                    legend_name2=None, line_attribute2=None,
                    **kwargs):
        """plot line(s)

        Parameters
        ----------
        x: a vector

        y: a vector, or a list of vectors

        y2: a vector, or a list of vectors whose values are displayed by yaxis2

        y2_title: str, title name of yaxis2

        legend_name: a name, or a list of names,
            must have same lenth as y

        line_attribute: a dict, or a list of dictionaries,
            must have same lenth as y
            eg. {'color': 'blue, 'dash': 'dot', 'width': 0.5}

        legend_name2: a name, or a list of names,
            must have same lenth as y

        line_attribute2: a dict, or a list of dictionaries,
            must have same lenth as y
            eg. {'color': 'blue, 'dash': 'dot', 'width': 0.5}

        **kwargs: a dictionary of additional params is passed in go.Scatter

        Returns
        -------
        A renderable plot
        """
        # make sure y is a list of vec
        if not isinstance(y, list):
            y = [y]
        if not isinstance(y2, list):
            y2 = [y2]
        # check if multiple vectors in y
        allowed_vector_type = \
            (list, pd.Series, np.ndarray, pd.DataFrame, pd.Index)
        if any(isinstance(vec, allowed_vector_type) for vec in y):
            # name
            if legend_name is not None:
                if not isinstance(legend_name, allowed_vector_type):
                    legend_name = [legend_name] * len(y)
                if len(legend_name) != len(y):
                    raise ValueError('length of legend_name must be the same as y') # noqa
            elif legend_name is None:
                legend_name = \
                    list(np.linspace(0, len(y)-1, len(y)).astype(int))
            # line
            if line_attribute is not None:
                if not isinstance(line_attribute, allowed_vector_type):
                    line_attribute = [line_attribute] * len(y)
                if len(line_attribute) != len(y):
                    raise ValueError('length of line_attribute must be the same as y') # noqa
            elif line_attribute is None:
                line_attribute = [{}] * len(y)

        # check if multiple vectors in y2
        if any(isinstance(vec, allowed_vector_type) for vec in y2):
            # name
            if legend_name2 is not None:
                if not isinstance(legend_name2, allowed_vector_type):
                    legend_name2 = [legend_name2] * len(y)
                if len(legend_name2) != len(y2):
                    raise ValueError('length of legend_name2 must be the same as y2') # noqa
            elif legend_name2 is None:
                legend_name2 = list(
                    np.linspace(0, len(y2)-1, len(y2)).astype(int)
                    + len(legend_name)
                )
            # line
            if line_attribute2 is not None:
                if not isinstance(line_attribute2, allowed_vector_type):
                    line_attribute2 = [line_attribute2] * len(y)
                if len(line_attribute2) != len(y2):
                    raise ValueError('length of line_attribute2 must be the same as y2') # noqa
            elif line_attribute2 is None:
                line_attribute2 = [{}] * len(y2)

            if y2_title is None:
                y2_title = 'y2'

            # store data
            data = []
            for i, vec in enumerate(y):
                data.append(
                    go.Scatter(
                        x=x,
                        y=vec,
                        name=legend_name[i],
                        line=line_attribute[i],
                        **kwargs
                    )
                )
            for i, vec in enumerate(y2):
                data.append(
                    go.Scatter(
                        x=x,
                        y=vec,
                        name=legend_name2[i],
                        line=line_attribute2[i],
                        **kwargs,
                        yaxis='y2'
                    )
                )

            layout = go.Layout(
                title=self.title,
                yaxis=self.yaxis,
                yaxis2=dict(
                    title=y2_title,
                    titlefont=dict(
                        color='rgb(148, 103, 189)'
                    ),
                    tickfont=dict(
                        color='rgb(148, 103, 189)'
                    ),
                    overlaying='y',
                    side='right'
                ),
                xaxis=self.xaxis,
                width=self.width,
                height=self.height,
                margin=go.Margin(
                    # l=50,
                    # r=50,
                    t=65,
                    b=60,
                    pad=4
                )
            )
            fig = go.Figure(data=data, layout=layout)
            plotly.offline.iplot(fig)

    def stack_barplot(self, x, y, legend_name=None, marker_attribute=None):
        """plot stacked bar chart(s)

        Parameters
        ----------
        x: a vector

        y: a vector, or a list of vectors

        legend_name: a name, or a list of names,
            must have same lenth as y

        marker_attribute: a dict, or a list of dictionaries,
            must have same lenth as y
            eg. {'color': 'red',
                 'line': {'color': 'red', 'width': 2}
                 }
        Returns
        -------
        A renderable plot
        """
        if not isinstance(y, list):
            y = [y]

        # check if multiple vectors in y
        allowed_vector_type = \
            (list, pd.Series, np.ndarray, pd.DataFrame, pd.Index)
        if any(isinstance(vec, allowed_vector_type) for vec in y):
            # name
            if legend_name is not None:
                if not isinstance(legend_name, allowed_vector_type):
                    legend_name = [legend_name] * len(y)
                if len(legend_name) != len(y):
                    raise ValueError('length of legend_name must be the same as y') # noqa
            elif legend_name is None:
                legend_name = \
                    list(np.linspace(0, len(y)-1, len(y)).astype(int))
            # marker
            if marker_attribute is not None:
                if not isinstance(marker_attribute, allowed_vector_type):
                    marker_attribute = [marker_attribute] * len(y)
                if len(marker_attribute) != len(y):
                    raise ValueError('length of marker_attribute must be the same as y') # noqa
            elif marker_attribute is None:
                marker_attribute = [{}] * len(y)

            # store data
            data = []
            for i, vec in enumerate(y):
                data.append(
                    go.Bar(
                        x=x,
                        y=vec,
                        name=legend_name[i],
                        marker=marker_attribute[i],
                        opacity=0.6
                    )
                )
            layout = go.Layout(
                title=self.title,
                yaxis=self.yaxis,
                xaxis=self.xaxis,
                width=self.width,
                height=self.height,
                margin=go.Margin(
                    # l=50,
                    # r=50,
                    t=65,
                    b=60,
                    pad=4
                ),
                barmode='stack'
            )
            fig = go.Figure(data=data, layout=layout)
            plotly.offline.iplot(fig)

    def barplot(self, x, y, text=None, marker_attribute=None, **kwargs):
        """plot bar chart

        Parameters
        ----------
        x: a vector of str

        y: a vector of values

        text: a list of str, eg. ['25%', '38%', ...]

        marker_attribute: a dict, or a list of dictionaries,
            must have same lenth as y
            eg. {'color': 'red',
                 'line': {'color': 'red', 'width': 2}
                 }

        **kwargs: a dictionary of additional params is passed in go.Bar

        Returns
        -------
        A renderable plot
        """
        if marker_attribute is None:
            marker_attribute = {}

        # store data
        data = [
            go.Bar(
                x=x,
                y=y,
                text=text,
                textposition='auto',
                marker=marker_attribute,
                opacity=0.6,
                **kwargs
            )
        ]
        layout = go.Layout(
            title=self.title,
            yaxis=self.yaxis,
            xaxis=self.xaxis,
            width=self.width,
            height=self.height,
            margin=go.Margin(
                # l=50,
                # r=50,
                t=65,
                b=85,
                pad=4
            )
        )
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def pieplot(self, labels, values, marker_attribute=None):
        """plot pie chart

        Parameters
        ----------
        labels: a vector of str

        values: a vector of values

        marker_attribute: a dict, or a list of dictionaries,
            must have same lenth as y
            eg. {'color': 'red',
                 'line': {'color': 'red', 'width': 2}
                 }
        Returns
        -------
        A renderable plot
        """
        if marker_attribute is None:
            marker_attribute = {}

        # store data
        data = [
            go.Pie(
                labels=labels,
                values=values,
                hoverinfo='label+value',
                textinfo='label+percent',
                textfont={'size': 10},
                marker=marker_attribute
            )
        ]
        layout = go.Layout(
            title=self.title,
            # yaxis=self.yaxis,
            # xaxis=self.xaxis,
            width=self.width,
            height=self.height,
            margin=go.Margin(
                # l=50,
                # r=50,
                t=65,
                b=85,
                pad=4
            )
        )
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def scatterplot(self, x, y, group=None, legend_name=None):
        """plot scatter plot

        Parameters
        ----------
        x: a vector, x is one column(feature) of a data

        y: a vector, y is another column(feature) of a data

        group: a vector, group is the label of data, it should have
            the same length as x, y

        legend_name: a name, or a dictionary of names, match the number of
            unique values in group
            eg. {
                    label_one: 'type one',
                    label_two: 'type two'
                }

        Returns
        -------
        A renderable plot
        """
        # check lenth
        x = force_array(x)
        y = force_array(y)
        check_consistent_length(x, y)
        if group is not None:
            group = force_array(group)
            check_consistent_length(x, y, group)
            # get all unique values from group
            unique_groups = np.unique(group)

            # check other args
            if legend_name is not None:
                check_consistent_length(unique_groups, list(legend_name))
            elif legend_name is None:
                legend_name = {v: v for v in unique_groups}

            # store data
            data = []
            for grp in unique_groups:
                data.append(
                    go.Scattergl(
                        x=x[group == grp],
                        y=y[group == grp],
                        name=legend_name[grp],
                        mode='markers'
                    )
                )
        elif group is None:
            trace = go.Scattergl(
                x=x,
                y=y,
                mode='markers'
            )
            data = [trace]

        layout = go.Layout(
            title=self.title,
            yaxis=self.yaxis,
            xaxis=self.xaxis,
            width=self.width,
            height=self.height,
            margin=go.Margin(
                # l=50,
                # r=50,
                t=65,
                b=60,
                pad=4)
        )
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)
