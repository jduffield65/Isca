import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from typing import List, Optional, Union


def label_subplots(fig: plt.Figure, ax_list: Union[plt.Axes, List[plt.Axes]], labels: Optional[List[str]] = None,
                   fontsize: float = 9):
    """
    This adds a label to each subplot in the top right corner.

    Args:
        fig: Figure containing subplots to add labels to.
        ax_list: [n_ax]
            List of all axes in the figure. If only one figure, can just provide the axes and not a list.
        labels: [n_ax]
            Label for each subplot. If not given, the label will just be the letters of the alphabet: a, b, c, ...
        fontsize: Font size to use
    """
    if isinstance(ax_list, plt.Axes):
        # If only provided one axes, make it into a list
        ax_list = [ax_list]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    if labels is None:
        labels = [f"{chr(ord('a') + i)})" for i in range(len(ax_list))]
    if len(labels) != len(ax_list):
        raise ValueError(f'{len(labels)} labels provided but there are {len(ax_list)} axes')
    for i, ax in enumerate(ax_list):
        ax.text(0.0, 1.0, labels[i], transform=ax.transAxes + trans,
                fontsize=fontsize, verticalalignment='top',
                bbox=dict(facecolor='1', edgecolor='none', pad=3.0))