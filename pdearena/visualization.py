# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def cleanup_axes(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.autoscale_view("tight")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)


def plot_scalar(ax, scalar: np.ndarray) -> mpl.image.AxesImage:
    """Plot a scalar field.

    Cleanups up the axes and returns the image object.

    Args:
        ax (mpl.axes.Axes): Axes to plot on.
        scalar (np.ndarray): 2D Scalar field to plot.

    Returns:
        (mpl.image.AxesImage): Image object.
    """
    im = ax.imshow(scalar)
    cleanup_axes(ax)
    return im


def plot_2dvec(ax, vec: np.ndarray) -> mpl.quiver.Quiver:
    """Plot a 2D vector field.

    Cleanups up the axes and returns the quiver object.

    Args:
        ax (mpl.axes.Axes): Axes to plot on.
        vec (np.ndarray): [2, ...] 2D Vector field to plot.

    Returns:
        (mpl.quiver.Quiver): Quiver object.
    """
    qv = ax.quiver(vec[0, ...], vec[1, ...])
    cleanup_axes(ax)
    return qv


def plot_scalar_sequence_comparison(init_field, ground_truth, prediction, fontsize=37, text_loc=(-10, 64)):
    """Plot a scalar field sequence comparison.

    Args:
        init_field (np.ndarray): Initial scalar field. We only plot the last time step of the initial field.
        ground_truth (np.ndarray): Ground truth scalar field.
        prediction (np.ndarray): Predicted scalar field.
        fontsize (int, optional): Fontsize for the text annotations. Defaults to 37.
        text_loc (tuple, optional): Location of the text annotations. Defaults to (-10, 64).

    Returns:
        (mpl.figure.Figure): Figure object.
    """
    assert ground_truth.shape == prediction.shape
    err = np.abs(ground_truth - prediction)
    n_timesteps = ground_truth.shape[0]
    scaling = max(ground_truth.shape[-1] / ground_truth.shape[-2], 1)
    fig = plt.figure(figsize=(n_timesteps * 6 * scaling, 15))

    # Plot the last timestep of init_field
    ## create space for init_field
    gs = GridSpec(3, n_timesteps + 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = plot_scalar(ax1, np.transpose(init_field[-1, ...], (1, 2, 0)))
    ax1.text(*text_loc, s="Ground truth", fontdict={"ha": "center", "va": "center"}, rotation=90, fontsize=fontsize)

    # Plot the ground truth sequence
    for i in range(n_timesteps):
        ax = fig.add_subplot(gs[0, i + 1])
        im1 = plot_scalar(ax, np.transpose(ground_truth[i, ...], (1, 2, 0)))

    # Plot the prediction sequence
    for i in range(n_timesteps):
        ax = fig.add_subplot(gs[1, i + 1])
        im2 = plot_scalar(ax, np.transpose(prediction[i, ...], (1, 2, 0)))
        if i == 0:
            ax.text(
                *text_loc, s="Prediction", fontdict={"ha": "center", "va": "center"}, rotation=90, fontsize=fontsize
            )

    # Plot the error sequence
    for i in range(n_timesteps):
        ax = fig.add_subplot(gs[2, i + 1])
        im3 = plot_scalar(ax, np.transpose(err[i, ...], (1, 2, 0)))
        if i == 0:
            ax.text(*text_loc, s="Error", fontdict={"ha": "center", "va": "center"}, rotation=90, fontsize=fontsize)

    # Plot the colorbars
    ## Create space for colorbars
    fig.subplots_adjust(wspace=0, hspace=0, right=0.9)
    ## Add colorbars
    cax1 = fig.add_axes([0.9, 0.15, 0.02, 0.2])
    fig.colorbar(im3, cax=cax1)
    cax1.tick_params(labelsize=fontsize * 0.6)
    cax2 = fig.add_axes([0.9, 0.4, 0.02, 0.2])
    fig.colorbar(im2, cax=cax2)
    cax2.tick_params(labelsize=fontsize * 0.6)

    return fig
