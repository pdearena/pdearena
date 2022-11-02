# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def tab30():
    """Create a discrete colormap with 30 unique colors. This colormap combines `matplotlib`'s
    `tab20b` and `tab20c` colormaps, removing the lightest color of each hue.

    Returns
    - cmap : `matplotlib.colors.ListedColormap`
    """
    colors = np.vstack([mpl.cm.tab20c.colors, mpl.cm.tab20b.colors])
    select_idx = np.repeat(np.arange(10), 3) * 4 + np.tile(np.arange(3), 10)
    return mpl.colors.ListedColormap(list(map(mpl.colors.to_hex, colors[select_idx])))


def tab40():
    """Create a discrete colormap with 40 unique colors.
    This colormap combines `matplotlib`'s `tab20b` and `tab20c` colormaps
    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
    """
    colors = np.vstack([mpl.cm.tab20c.colors, mpl.cm.tab20b.colors])
    return mpl.colors.ListedColormap(colors)


def cleanup_axes(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.autoscale_view("tight")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)


def plot_scalar(ax, scalar: np.ndarray) -> None:
    im = ax.imshow(scalar)
    cleanup_axes(ax)
    return im


def plot_2dvec(ax, vec: np.ndarray) -> None:
    ax.quiver(vec[0, ...], vec[1, ...])
    cleanup_axes(ax)


# https://stackoverflow.com/questions/68690442/plotting-3d-vector-field-in-python
def plot_3d_quiver(ax, x, y, z, u, v, w):
    # COMPUTE LENGTH OF VECTOR -> MAGNITUDE
    c = np.sqrt(np.abs(v) ** 2 + np.abs(u) ** 2 + np.abs(w) ** 2)

    c = (c.ravel() - c.min()) / c.ptp()
    # Repeat for each body line and two head lines
    c = np.concatenate((c, np.repeat(c, 2)))
    # Colormap
    c = plt.cm.jet(c)

    ax.quiver(x, y, z, u, v, w, colors=c, length=0.5, arrow_length_ratio=0.2)
    ax.axis("off")
    plt.gca().invert_zaxis()
