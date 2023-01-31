import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import ipywidgets as widgets
from typing import Union, Optional
from .funcs import in_notebook


# TODO: plot_segmentation fails if only one image

def view_stack_tk(frames: Union[list, np.ndarray], start_frame: int = 0, end_frame: Optional[int] = None,
                  show: bool = True):
    """
    Interactive stack viewer

    Args:
        frames: either a numpy array of a 2D image or a list of 2D arrays
        start_frame: optional. If speficied only show frames after this index
        end_frame: optional. If specified only show frames before this index
        show: if True, show the image

    Returns:
        figure and axis

    """

    # Detect if single frame or stack
    if type(frames) is list:
        stack = True
        frames_ = frames
    elif len(frames.shape) == 3:
        stack = True
        frames_ = list(frames)
    else:
        stack = False
        frames_ = [frames, ]

    # Set up figure
    plt.ion()
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Specify ylim
    vmax = max([np.percentile(i, 99.9) for i in frames_])
    vmin = min([np.percentile(i, 0.1) for i in frames_])

    # Stack
    if stack:
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        if end_frame is None:
            end_frame = len(frames_)
        sframe = Slider(axframe, 'Frame', start_frame, end_frame, valinit=start_frame, valfmt='%d')

        def update(i):
            ax.clear()
            ax.imshow(frames_[int(i)], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

        sframe.on_changed(update)
        update(start_frame)

    # Single frame
    else:
        ax.imshow(frames_[0], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.canvas.set_window_title('')

    if show:
        plt.show(block=True)

    return fig, ax


def view_stack_jupyter(frames: Union[list, np.ndarray], start_frame: int = 0, end_frame: Optional[int] = None):
    # Detect if single frame or stack
    if type(frames) is list:
        stack = True
        frames_ = frames
    elif len(frames.shape) == 3:
        stack = True
        frames_ = list(frames)
    else:
        stack = False
        frames_ = [frames, ]

    # Set up figure
    fig, ax = plt.subplots()

    # Specify ylim
    vmax = max([np.percentile(i, 99.9) for i in frames_])
    vmin = min([np.percentile(i, 0.1) for i in frames_])

    # Stack
    if stack:
        if end_frame is None:
            end_frame = len(frames_) - 1

        @widgets.interact(Frame=(start_frame, end_frame, 1))
        def update(Frame=start_frame):
            ax.clear()
            ax.imshow(frames_[int(Frame)], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

    # Single frame
    else:
        ax.imshow(frames_[0], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.set_size_inches(4, 4)
    fig.tight_layout()

    return fig, ax


def view_stack(frames: Union[list, np.ndarray], start_frame: int = 0, end_frame: Optional[int] = None):
    jupyter = in_notebook()
    if jupyter:
        view_stack_jupyter(frames, start_frame, end_frame)
    else:
        view_stack_tk(frames, start_frame, end_frame)


def plot_segmentation(frames: Union[list, np.ndarray], rois: Union[list, np.ndarray]):
    """
    Plot segmentation results

    Args:
        frames: either a numpy array of a 2D image or a list of 2D arrays
        rois: either a single two-column numpy array of ROI coordinates or a list of arrays

    Returns:
        figure and axis

    """

    fig, ax = plt.subplots()

    # Detect if single frame or stack
    if type(frames) is list:
        stack = True
        frames_ = frames
    elif len(frames.shape) == 3:
        stack = True
        frames_ = list(frames)
    else:
        stack = False
        frames_ = [frames, ]

    # Specify ylim
    ylim_top = max([np.percentile(i, 99.9) for i in frames_])
    ylim_bottom = min([np.percentile(i, 0.01) for i in frames_])

    # Single frame
    if not stack:
        ax.imshow(frames_[0], cmap='gray', vmin=ylim_bottom, vmax=ylim_top)
        ax.plot(rois[:, 0], rois[:, 1], c='lime')
        ax.scatter(rois[0, 0], rois[0, 1], c='lime')
        ax.set_xticks([])
        ax.set_yticks([])

    # Stack
    else:

        # Add frame slider
        plt.subplots_adjust(left=0.25, bottom=0.25)
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        sframe = Slider(axframe, 'Frame', 0, len(frames_), valinit=0, valfmt='%d')

        def update(i):
            ax.clear()
            ax.imshow(frames_[int(i)], cmap='gray', vmin=ylim_bottom, vmax=ylim_top)
            ax.plot(rois[int(i)][:, 0], rois[int(i)][:, 1], c='lime')
            ax.scatter(rois[int(i)][0, 0], rois[int(i)][0, 1], c='lime')
            ax.set_xticks([])
            ax.set_yticks([])

        sframe.on_changed(update)
        update(0)

    fig.canvas.set_window_title('Segmentation')
    plt.show(block=True)

    return fig, ax


def plot_segmentation_jupyter(frames: Union[list, np.ndarray], rois: Union[list, np.ndarray]):
    """
    Plot segmentation results - use this function in a jupyter notebook environment

    Args:
        frames: either a numpy array of a 2D image or a list of 2D arrays
        rois: either a single two-column numpy array of ROI coordinates or a list of arrays

    Returns:
        figure and axis

    """

    fig, ax = plt.subplots()

    # Detect if single frame or stack
    if type(frames) is list:
        stack = True
        frames_ = frames
    elif len(frames.shape) == 3:
        stack = True
        frames_ = list(frames)
    else:
        stack = False
        frames_ = [frames, ]

    # Specify ylim
    ylim_top = max([np.percentile(i, 99.9) for i in frames_])
    ylim_bottom = min([np.percentile(i, 0.01) for i in frames_])

    # Single frame
    if not stack:
        ax.imshow(frames_[0], cmap='gray', vmin=ylim_bottom, vmax=ylim_top)
        ax.plot(rois[:, 0], rois[:, 1], c='lime')
        ax.scatter(rois[0, 0], rois[0, 1], c='lime')
        ax.set_xticks([])
        ax.set_yticks([])

    # Stack
    else:
        @widgets.interact(Frame=(0, len(frames_) - 1, 1))
        def update(Frame=0):
            ax.clear()
            ax.imshow(frames_[int(Frame)], cmap='gray', vmin=ylim_bottom, vmax=ylim_top)
            ax.plot(rois[int(Frame)][:, 0], rois[int(Frame)][:, 1], c='lime')
            ax.scatter(rois[int(Frame)][0, 0], rois[int(Frame)][0, 1], c='lime')
            ax.set_xticks([])
            ax.set_yticks([])

    fig.set_size_inches(4, 4)
    fig.tight_layout()

    return fig, ax


def plot_quantification(mems: Union[list, np.ndarray]):
    """
    Plot quantification results

    Args:
        mems: either a numpy array of membrane concentrations for one image or a list of arrays for multiple images

    Returns:
        figure and axis

    """
    fig, ax = plt.subplots()

    # Detect if single frame or stack
    if type(mems) is list:
        stack = True
        mems_ = mems
    elif len(mems.shape) == 2:
        stack = True
        mems_ = list(mems)
    else:
        stack = False
        mems_ = [mems, ]

    # Single frame
    if not stack:
        ax.plot(mems_[0])
        ax.set_xlabel('Position')
        ax.set_ylabel('Membrane concentration')
        ax.set_ylim(bottom=min(0, np.min(mems_[0])))
        ax.axhline(0, c='k', linestyle='--')

    # Stack
    else:

        # Specify ylim
        ylim_top = max([np.max(m) for m in mems_])
        ylim_bottom = min([np.min(m) for m in mems_])

        # Add frame silder
        plt.subplots_adjust(left=0.25, bottom=0.25)
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        sframe = Slider(axframe, 'Frame', 0, len(mems_), valinit=0, valfmt='%d')

        def update(i):
            ax.clear()
            ax.plot(mems_[int(i)])
            ax.axhline(0, c='k', linestyle='--')
            ax.set_xlabel('Position')
            ax.set_ylabel('Membrane concentration')
            ax.set_ylim(min(ylim_bottom, 0), ylim_top)

        sframe.on_changed(update)
        update(0)

    fig.canvas.set_window_title('Membrane Quantification')
    plt.show(block=True)

    return fig, ax


def plot_quantification_jupyter(mems: Union[list, np.ndarray]):
    """
    Plot quantification results - use this function in a jupyter notebook environment

    Args:
        mems: either a numpy array of membrane concentrations for one image or a list of arrays for multiple images

    Returns:
        figure and axis

    """
    fig, ax = plt.subplots()

    # Detect if single frame or stack
    if type(mems) is list:
        stack = True
        mems_ = mems
    elif len(mems.shape) == 2:
        stack = True
        mems_ = list(mems)
    else:
        stack = False
        mems_ = [mems, ]

    # Single frame
    if not stack:
        ax.plot(mems_[0])
        ax.set_xlabel('Position')
        ax.set_ylabel('Membrane concentration')
        ax.set_ylim(bottom=min(0, np.min(mems_[0])))
        ax.axhline(0, c='k', linestyle='--')

    # Stack
    else:

        # Specify ylim
        ylim_top = max([np.max(m) for m in mems_])
        ylim_bottom = min([np.min(m) for m in mems_] + [0])

        @widgets.interact(Frame=(0, len(mems_) - 1, 1))
        def update(Frame=0):
            ax.clear()
            ax.plot(mems_[int(Frame)])
            ax.axhline(0, c='k', linestyle='--')
            ax.set_xlabel('Position')
            ax.set_ylabel('Membrane concentration')
            ax.set_ylim(ylim_bottom, ylim_top)

    fig.set_size_inches(5, 3)
    fig.tight_layout()

    return fig, ax


class FitPlotter:
    def __init__(self,
                 target: Union[list, np.ndarray],
                 fit: Union[list, np.ndarray]):

        # Detect if single frame or stack
        if type(target) is list:
            self.stack = True
            target_ = target
            fit_ = fit
        elif len(target.shape) == 3:
            self.stack = True
            target_ = list(target)
            fit_ = list(fit)
        else:
            self.stack = False
            target_ = [target, ]
            fit_ = [fit, ]

        # Internal variables
        self.target = target_
        self.fit = fit_
        self.pos = 10

        # Set up figure
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(3, 3)
        self.ax1 = self.fig.add_subplot(gs[0, :])
        self.ax2 = self.fig.add_subplot(gs[1:, :])

        # Specify ylim
        straight_max = max([np.max(i) for i in self.target])
        straight_min = min([np.min(i) for i in self.target])
        fit_max = max([np.max(i) for i in self.fit])
        fit_min = min([np.min(i) for i in self.fit])
        self.ylim_top = max([straight_max, fit_max])
        self.ylim_bottom = min([straight_min, fit_min])

        # Frame slider
        if self.stack:
            plt.subplots_adjust(bottom=0.25, left=0.25)
            axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
            slider_frame = Slider(axframe, 'Frame', 0, len(self.target), valinit=0, valfmt='%d')
            slider_frame.on_changed(lambda f: self.update_frame(int(f)))

        # Initial plot
        self.update_frame(0)

        # Show
        self.fig.canvas.set_window_title('Local fits')
        plt.show(block=True)

    def update_pos(self, p: float):
        self.pos = int(p)
        self.ax1_update()
        self.ax2_update()

    def update_frame(self, i: int):
        self._target = self.target[i]
        self._fit = self.fit[i]

        # Position slider
        self.slider_pos = Slider(self.ax1, '', 0, len(self._target[0, :]), valinit=self.pos, valfmt='%d',
                                 facecolor='none', edgecolor='none')
        self.slider_pos.on_changed(self.update_pos)

        self.ax1_update()
        self.ax2_update()

    def ax1_update(self):
        self.ax1.clear()
        self.ax1.imshow(self._target, cmap='gray', vmin=self.ylim_bottom, vmax=1.1 * self.ylim_top)
        self.ax1.axvline(self.pos, c='r')
        self.ax1.set_yticks([])
        self.ax1.set_xlabel('Position')
        self.ax1.xaxis.set_label_position('top')

    def ax2_update(self):
        self.ax2.clear()
        self.ax2.plot(self._target[:, self.pos], label='Actual')
        self.ax2.plot(self._fit[:, self.pos], label='Fit')
        self.ax2.set_xticks([])
        self.ax2.set_ylabel('Intensity')
        self.ax2.legend(frameon=False, loc='upper left', fontsize='small')
        self.ax2.set_ylim(bottom=self.ylim_bottom, top=self.ylim_top)


def plot_fits(target: Union[list, np.ndarray], fit_total: Union[list, np.ndarray]):
    fp = FitPlotter(target, fit_total)
    return fp.fig, (fp.ax1, fp.ax2)


def plot_fits_jupyter(target: Union[list, np.ndarray], fit: Union[list, np.ndarray]):
    # Detect if single frame or stack
    if type(target) is list:
        stack = True
        target = target
        fit = fit
    elif len(target.shape) == 3:
        stack = True
        target = list(target)
        fit = list(fit)
    else:
        stack = False
        target = [target, ]
        fit = [fit, ]

    # Set up figure
    fig = plt.figure()
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1:, :])

    # Specify ylim
    straight_max = max([np.max(i) for i in target])
    straight_min = min([np.min(i) for i in target])
    fit_max = max([np.max(i) for i in fit])
    fit_min = min([np.min(i) for i in fit])
    ylim_top = max([straight_max, fit_max])
    ylim_bottom = min([straight_min, fit_min])

    if stack:

        @widgets.interact(Frame=(0, len(target) - 1, 1), Position=(0, 1, 0.01))
        def update(Frame: int = 0, Position: float = 0.1):
            position = int(Position * target[int(Frame)].shape[1] - 1)

            ax1.clear()
            ax1.imshow(target[int(Frame)], cmap='gray', vmin=ylim_bottom, vmax=1.1 * ylim_top)
            ax1.axvline(position, c='r')
            ax1.set_yticks([])
            ax1.set_xlabel('Position')
            ax1.xaxis.set_label_position('top')

            ax2.clear()
            ax2.plot(target[int(Frame)][:, position], label='Actual')
            ax2.plot(fit[int(Frame)][:, position], label='Fit')
            ax2.set_xticks([])
            ax2.set_ylabel('Intensity')
            ax2.legend(frameon=False, loc='upper left', fontsize='small')
            ax2.set_ylim(bottom=ylim_bottom, top=ylim_top)

    else:

        @widgets.interact(Position=(0, 1, 0.01))
        def update(Position: float = 0.1):
            position = int(Position * (target[0].shape[1] - 1))

            ax1.clear()
            ax1.imshow(target[0], cmap='gray', vmin=ylim_bottom, vmax=1.1 * ylim_top)
            ax1.axvline(position, c='r')
            ax1.set_yticks([])
            ax1.set_xlabel('Position')
            ax1.xaxis.set_label_position('top')

            ax2.clear()
            ax2.plot(target[0][:, position], label='Actual')
            ax2.plot(fit[0][:, position], label='Fit')
            ax2.set_xticks([])
            ax2.set_ylabel('Intensity')
            ax2.legend(frameon=False, loc='upper left', fontsize='small')
            ax2.set_ylim(bottom=ylim_bottom, top=ylim_top)

    fig.set_size_inches(5, 3)
    fig.tight_layout()

    return fig, (ax1, ax2)
