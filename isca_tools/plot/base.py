import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.collections import LineCollection
from typing import List, Optional, Union
import warnings

import numpy as np


def label_subplots(fig: plt.Figure, ax_list: Union[plt.Axes, List[plt.Axes]], labels: Optional[List[str]] = None,
                   fontsize: float = 9, fontcolor: str = 'k', box_alpha: float = 1, pos_x=10, pos_y=-5):
    """
    This adds a label to each subplot in the top right corner.

    Args:
        fig: Figure containing subplots to add labels to.
        ax_list: [n_ax]
            List of all axes in the figure. If only one figure, can just provide the axes and not a list.
        labels: [n_ax]
            Label for each subplot. If not given, the label will just be the letters of the alphabet: a, b, c, ...
        fontsize: Font size to use
        fontcolor: What color font to use
        box_alpha: Opacity of box bounding text (1 is opaque and 0 is transparent)
        pos_x: Can specify distance from top right corner.
        pos_y: Can specify distance from top right corner (should be negative).
    """
    if isinstance(ax_list, plt.Axes):
        # If only provided one axes, make it into a list
        ax_list = [ax_list]
    trans = mtransforms.ScaledTranslation(pos_x/72, pos_y/72, fig.dpi_scale_trans)
    if labels is None:
        labels = [f"{chr(ord('a') + i)})" for i in range(len(ax_list))]
    if len(labels) != len(ax_list):
        raise ValueError(f'{len(labels)} labels provided but there are {len(ax_list)} axes')
    for i, ax in enumerate(ax_list):
        ax.text(0.0, 1.0, labels[i], transform=ax.transAxes + trans,
                fontsize=fontsize, verticalalignment='top', color=fontcolor,
                bbox=dict(facecolor='1', edgecolor='none', pad=3.0, alpha=box_alpha))


def colored_line(x: np.ndarray, y: np.ndarray, c: np.ndarray, ax: plt.Axes, **lc_kwargs) -> LineCollection:
    """
    Plot a line with a color specified along the line by a third value.
    Code copied from
    [matplotlib website](https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html).

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Args:
        x: Horizontal coordinates of data points
        y: Vertical coordinates of data points
        c: The color values, which should be the same size as `x` and `y`.
        ax: Axis object on which to plot the colored line.
        **lc_kwargs:
            Any additional arguments to pass to matplotlib.collections.LineCollection
            constructor. This should not include the array keyword argument because
            that is set to the color argument. If provided, it will be overridden.

    Returns:
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)