import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.collections import LineCollection
from typing import List, Optional, Union, Tuple
import warnings
import numpy as np
import os


def savefig(fig: plt.Figure, file_name: str = 'output', output_dir: str = '/Users/joshduffield/Desktop/',
            format: str = 'pdf', dpi: Union[float, str] = 800, bbox_inches: Optional[str] = 'tight',
            pad_inches: Union[float, str] = 0.05, overwrite_file: bool = False, save_if_exists: bool = True) -> None:
    """
    Function to save figure, basically just calls
    [`plt.savefig`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html) but has more useful
    default values, and option to not overwrite figure if already exists.

    Args:
        fig: Matplolib figure that you would like to save
        file_name: Name of saved figure file in `output_dir`. Should not include `format`.
        output_dir: Directory into which figure file will be saved.
        format: The file format, e.g. `'png'`, `'pdf'`, `'svg'`, `'jpeg'`.</br>
            If a different format is included in `file_name`, then the format in `file_name` will be used.
        dpi: The resolution in dots per inch. If `'figure'`, use the figure's dpi value.
        bbox_inches: Bounding box in inches: only the given portion of the figure is saved.
            If `'tight'`, try to figure out the tight bbox of the figure.
        pad_inches: Amount of padding in inches around the figure when bbox_inches is 'tight'.
            If `'layout'` use the padding from the constrained or compressed layout engine;
            ignored if one of those engines is not in use.
        overwrite_file: If `False` and file already exists, will add an integer (starting with `2`) to end of
            `file_name` until find a name that does not exist. Only relevant if `save_if_exists=True`.
        save_if_exists: If `False` and file already exists, will not save any file, otherwise will save according
            to `overwrite_file`.

    """
    format_array = ['png', 'pdf', 'svg', 'jpeg']
    if format not in format_array:
        raise ValueError(f'Format {format} is not supported, must be in {format_array}')

    # If file_name contains a different format to `format`, use format in file_name
    format_with_dot = '.' + format.replace(".", "")
    for key in format_array:
        if f".{key}" in file_name:
            format_with_dot = f".{key}"

    # Check to ensure don't have the format included twice
    if format_with_dot in file_name:
        file_name_use = file_name.replace(format_with_dot, '')
    else:
        file_name_use = file_name
    file_path = os.path.join(output_dir, file_name_use + format_with_dot)
    if os.path.isfile(file_path) and not save_if_exists:
        pass
    else:
        if not overwrite_file:
            # Add number to file name so does not overwrite existing file
            i = 1
            while os.path.isfile(file_path):
                i += 1
                file_path = os.path.join(output_dir, file_name_use + f'{i}{format_with_dot}')
        fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    return None


def label_subplots(fig: plt.Figure, ax_list: Union[plt.Axes, List[plt.Axes]], labels: Optional[List[str]] = None,
                   fontsize: float = 9, fontcolor: str = 'k', box_alpha: float = 1,
                   pos_x: float=5, pos_y: float=-5) -> None:
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
    trans = mtransforms.ScaledTranslation(pos_x / 72, pos_y / 72, fig.dpi_scale_trans)
    if labels is None:
        labels = [f"{chr(ord('a') + i)})" for i in range(len(ax_list))]
    if len(labels) != len(ax_list):
        raise ValueError(f'{len(labels)} labels provided but there are {len(ax_list)} axes')
    for i, ax in enumerate(ax_list):
        ax.text(0.0, 1.0, labels[i], transform=ax.transAxes + trans,
                fontsize=fontsize, verticalalignment='top', color=fontcolor,
                bbox=dict(facecolor='1', edgecolor='none', pad=3.0, alpha=box_alpha))
    return None


def get_fig_n_rows_cols(fig: plt.Figure) -> Tuple[int, int]:
    """
    Returns the number of rows and columns of subplots in a figure.

    Args:
        fig: Figure to find subplot arrangement for.

    Returns:
        n_row: Number of rows of subplots.
        n_col: Number of columns of subplots.
    """
    # Get the axes objects from the figure
    axes = fig.axes

    # To determine the number of rows and columns, we need to find the grid shape
    # by considering the number of axes and their positions

    # Sort axes by their x and y positions to infer the grid structure
    sorted_axes = sorted(axes, key=lambda ax: (ax.get_position().y0, ax.get_position().x0))

    # Assuming the grid is arranged row-wise (top-to-bottom, left-to-right)
    # Calculate number of rows and columns based on sorted axes positions
    n_col = len(set(ax.get_position().x0 for ax in sorted_axes))  # Number of unique x positions
    n_row = len(set(ax.get_position().y0 for ax in sorted_axes))  # Number of unique y positions
    return n_row, n_col


def fig_resize(fig: plt.Figure, width_fig_desired: Optional[float]=None, ar: float=4/3) -> None:
    """
    Change height of figure such that aspect ratio of each subplot is set to `ar`, while the width is maintained
    at the same value.

    Args:
        fig: Figure to resize.
        width_fig_desired: Width of figure after resize in inches. If not provided, will keep current width.
        ar: Desired aspect ratio (width/height) of each subplot within `fig`.

    """
    if width_fig_desired is None:
        width_fig_desired = fig.get_size_inches()[0]
    n_row, n_col = get_fig_n_rows_cols(fig)
    width_subplot_desired = (width_fig_desired / n_col)
    height_subplot_desired = width_subplot_desired / ar  # height of each subplot
    height_fig_desired = height_subplot_desired * n_row
    fig.set_size_inches(width_fig_desired, height_fig_desired)
    return None


def update_fontsize(fig: plt.Figure, base_fontsize: float=8, base_ax_width: float=2.464) -> float:
    """
    Resize fontsize based on subplot width, given that `base_fontsize` is a good fontsize for a subplot
    of width `base_ax_width` inches.

    Args:
        fig: Figure to change fontsize for.
        base_fontsize: A good fontsize for a subplot of width `base_ax_width` inches.
        base_ax_width: A subplot of width `base_ax_width` inches, looks good with fontsize set to `base_fontsize`.

    Returns:
        new_fontsize: The new fontsize
    """
    ax_width = fig.axes[0].get_position().width * fig.get_size_inches()[0]      # use first axes to get subplot width
    scale_factor = ax_width/base_ax_width
    new_fontsize = scale_factor * base_fontsize
    for text in fig.findobj(plt.Text):  # Find all text objects
        text.set_fontsize(new_fontsize)
    return new_fontsize


def update_linewidth(fig: plt.Figure, base_linewidth: float=1, base_ax_width: float=2.464) -> None:
    """
    Resize linewidths based on subplot width, given that `base_linewidth` is a good linewidth for a subplot
    of width `base_ax_width` inches.

    Args:
        fig: Figure to change linewidth for.
        base_linewidth: A good linewidth for a subplot of width `base_ax_width` inches.
        base_ax_width: A subplot of width `base_ax_width` inches, looks good with linewidth set to `base_linewidth`.

    """
    ax_width = fig.axes[0].get_position().width * fig.get_size_inches()[0]  # use first axes to get subplot width
    scale_factor = ax_width / base_ax_width
    new_linewidth = base_linewidth * scale_factor

    # Set new line width for the plot
    for ax in fig.axes:
        for line in ax.get_lines():
            line.set_linewidth(new_linewidth)
    return None


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
    ax.add_collection(lc)
    ax.autoscale_view()         # update axis limits
    return lc
