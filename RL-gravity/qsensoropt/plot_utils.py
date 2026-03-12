"""Module containing some useful routines
for plotting the data contained in the
csv file produced by the routines of the
:py:mod:`~.utils` module. This module relies on
the functionalities of the Matplotlib.
"""

from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import gaussian_kde
from numpy import vstack
from os.path import join


def plot_2_values(
    df: DataFrame,
    x: str, y: str,
    figsize: Tuple[float, float] = (3.0, 2.0),
    dpi: float = 200.0,
    path: Optional[str] = None,
    log_scale: bool = True,
    log_scale_x: bool = False,
    vertical: bool = True,
):
    """Plots two arrays of values,
    in a standard 2d line plot.

    Parameters
    ----------
    df: :py:obj:`DataFrame`
        Pandas `DataFrame` containing the values to plot.
        It must have two columns named `x` and `y`
        respectively. This routine will plot
        `df[y]` as a function of `df[x]`.
    x: str
        Name of the x-axis
    y: str
        Name of the y-axis.
    figsize: Tuple[float, float] = (3.0, 2.0)
        Width and height of the figure in inches.
    dpi: float = 200.0
        Resolution of the figure in dots-per-inch.
    path: str = None
        Directory in which the figure
        is saved. If it is not passed, the figure
        is not saved on disk.
    log_scale: bool = True
        Logarithmic scale on the y-axis.
    log_scale_x: bool = False
        Logarithmic scale on the x-axis.
    vertical: bool = True
        Orientation of the label
        of the y-axis.

    Notes
    -----
    The name of the saved plot is
    `plot_{x}_{y}.png`.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlabel(x)
    ax.set_ylabel(
        y,
        rotation='vertical' if
        vertical else 'horizontal',
    )
    if log_scale:
        ax.set_yscale('log')
    if log_scale_x:
        ax.set_xscale('log')
    ax.minorticks_on()
    ax.grid(color="gray")
    ax.plot(df[x], df[y])
    plot_name = f"plot_{x}_{y}"
    fig.tight_layout()
    if path is not None:
        fig.savefig(join(path, plot_name)+".png")
    fig.show()

def plot_multiples(
    list_df: List[DataFrame],
    x: str, y: str,
    list_labels: List[str],
    figsize: Tuple[float, float] = (3.0, 2.0),
    dpi: float = 200.0,
    path: Optional[str] = None,
    log_scale: bool = True,
    log_scale_x: bool = False,
    title: str = "",
    legend_location: str = "upper right",
    vertical: bool = True,
    list_line_syles: List[str] = None,
    lower_bound_df: DataFrame = None,
):
    """Plots two arrays of values for multiple
    datasets, for comparison purposes, in a
    standard line plot.

    Parameters
    ----------
    list_df: List[:py:obj:`DataFrame`]
        List of Pandas `DataFrame` containing the
        values to plot.
        Each `DataFrame` in the list must have
        two columns named `x` and `y`
        respectively. This routine will plot
        `df[y]` as a function of `df[x]`
        for all the elements `df` of `list_df`
        on the same figure.
    x: str
        Name of the x-axis
    y: str
        Name of the y-axis.
    list_labels: List[str]
        List of labels for each `DataFrame`
        in `list_df`. These are the names
        visualized in the legend.
    figsize: Tuple[float, float] = (3.0, 2.0)
        Width and height of the figure in inches.
    dpi: float = 200.0
        Resolution of the figure in dots-per-inch.
    path: str = None
        Directory in which the figure
        is saved. If it is not passed, the figure
        is not saved on disk.
    log_scale: bool = True
        Logarithmic scale on the y-axis.
    log_scale_x: bool = False
        Logarithmic scale on the x-axis.
    title: str = ""
        Title of the plot.
    legend_location: str = "upper right",
        Location of the legend.
    vertical: bool = True
        Orientation of the label
        of the y-axis.
    list_line_syles: List[str] = None
        List of strings specifying the style
        for each line.
    lower_bound_df: DataFrame = None
        DataFrame containing the lower bound
        for the precision plot. Under this line
        everything must be colored in grey.

    Notes
    -----
    The name of the saved plot is
    `plot_{x}_{y}.png`.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes()
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(
        y, rotation='vertical' if
        vertical else 'horizontal',
    )
    if list_line_syles is None:
        list_line_syles = ['solid' for _ in list_df]
    if log_scale:
        ax.set_yscale("log")
    if log_scale_x:
        ax.set_xscale('log')
    ax.minorticks_on()
    ax.grid(color="gray")
    for prec_df, label, style in \
      zip(list_df, list_labels, list_line_syles):
        ax.plot(
            prec_df[x], prec_df[y], linestyle=style, label=label,
        )
    ax.legend(loc=legend_location, prop={'size': 5})
    # Adding the lower bound
    if lower_bound_df is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_autoscale_on(False)
        ax.fill_between(
            lower_bound_df[x].values, lower_bound_df[y].values,
            color='lightgray',
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    plot_name = f"plot_{x}_{y}"
    fig.tight_layout()
    if path is not None:
        fig.savefig(join(path, plot_name)+".png")
    fig.show()

def plot_3_values(
    df: DataFrame,
    x: str, y: str, color: str,
    figsize: Tuple[float, float] = (3.0, 2.0),
    dpi: float = 200.0,
    path: Optional[str] = None,
    vertical: bool = True,
):
    """Plots three arrays of values,
    one encoded in the color of the
    point in a scatter plot.

    Parameters
    ----------
    df: :py:obj:`DataFrame`
        Pandas `DataFrame` containing the values to plot.
        It must have three columns named respectively
        `x`, `y`, and `color`.
        This routine will plot the tuples of points
        (`df[x]`, `df[y]`) in a scatter plot. The
        color of the points is the column
        `df[color]`, which must have values in `[0, 1]`.
    x: str
        Name of the x-axis
    y: str
        Name of the y-axis.
    color: str
        Name of the column represented as the
        color of points in the scatter plot.
    figsize: Tuple[float, float] = (3.0, 2.0)
        Width and height of the figure in inches.
    dpi: float = 200.0
        Resolution of the figure in dots-per-inch.
    path: str = None
        Directory in which the figure
        is saved. If it is not passed, the figure
        is not saved on disk.
    vertical: bool = True
        Orientation of the label
        of the y-axis.

    Notes
    -----
    The name of the saved plot is
    `plot_{x}_{y}_c_{color}.png`.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.cm.get_cmap('plasma')
    ax.set_xlabel(x)
    ax.set_ylabel(
        y,
        rotation='vertical' if
        vertical else 'horizontal',
    )
    ax.minorticks_on()
    ax.set_facecolor("gainsboro")
    ax.scatter(df[x], df[y], c=cmap(df[color]), s=5, edgecolor=['none'])
    plot_name = f"scatter_{x}_{y}_c_{color}"
    fig.tight_layout()
    if path is not None:
        fig.savefig(join(path, plot_name)+".png")
    fig.show()


def plot_4_values(
    df: DataFrame,
    x: str, y: str, z: str, color: str,
    figsize: Tuple[float, float] = (6.0, 4.0),
    dpi: float = 300.0,
    path: Optional[str] = None,
):
    """Plot four array of values in a 3d scatter
    plot. One of the values is encoded in the color
    of the points.

    Parameters
    ----------
    df: :py:obj:`DataFrame`
        Pandas `DataFrame` containing the values to plot.
        It must have four columns named respectively
        `x`, `y`, `z`, and `color`.
        This routine will plot the tuples of points
        (`df[x]`, `df[y]`, `df[z]`) in a 3d scatter plot. The
        color of the points is the column
        `df[color]`, which must have values in `[0, 1]`.
    x: str
        Name of the x-axis
    y: str
        Name of the y-axis.
    z: str
        Name of the z-axis.
    color: str
        Name of the column represented as the
        color of points in the scatter plot.
    figsize: Tuple[float, float] = (6.0, 4.0)
        Width and height of the figure in inches.
    dpi: float = 300.0
        Resolution of the figure in dots-per-inch.
    path: str = None
        Directory in which the figure
        is saved. If it is not passed, the figure
        is not saved on disk.

    Notes
    -----
    The name of the saved plot is
    `plot_{x}_{y}_{z}_c_{color}.png`.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    cmap = plt.cm.get_cmap('plasma')
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(z)
    ax.minorticks_on()
    ax.scatter(
        df[x], df[y], df[z], c=cmap(df[color]), s=5,
        edgecolor=['none'],
    )
    plot_name = f"scatter_{x}_{y}_{z}_c_{color}"
    fig.tight_layout()
    if path is not None:
        fig.savefig(join(path, plot_name)+".png")
    fig.show()
