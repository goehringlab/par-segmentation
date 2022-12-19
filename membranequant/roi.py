import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import splprep, splev, interp1d
from matplotlib.widgets import Button
import ipywidgets as widgets
from typing import Union, Optional
from matplotlib.backend_bases import MouseEvent, KeyEvent

"""
Todo: This no longer works with multiple channels - intensity ranges
Todo: Ability to specify a directory and open all channels. Or an nd file

"""


def def_roi(stack: Union[np.ndarray, list], spline: bool = True, start_frame: int = 0, end_frame: Optional[int] = None,
            periodic: bool = True, show_fit: bool = True, k: int = 3):
    r = ROI(stack, spline=spline, start_frame=start_frame, end_frame=end_frame, periodic=periodic, show_fit=show_fit,
            k=k)
    r.run()
    return r.roi


class ROI:
    """
    Instructions:
    - click to lay down points
    - backspace at any time to remove last point
    - press enter to select area (if spline=True will fit spline to points, otherwise will fit straight lines)
    - at this point can press backspace to go back to laying points
    - press enter again to close and return ROI

    :param img: input image
    :param spline: if true, fits spline to inputted coordinates
    :return: cell boundary coordinates
    """

    def __init__(self,
                 img: Union[np.ndarray, list],
                 spline: bool = True,
                 start_frame: int = 0,
                 end_frame: Optional[int] = None,
                 periodic: bool = True,
                 show_fit: bool = True,
                 k: int = 3):

        # Detect if single frame or stack
        if type(img) is list:
            self.img_type = 'list'
            self.images = img

        elif len(img.shape) == 3:
            self.img_type = 'stack'
            self.images = list(img)
        else:
            self.img_type = 'single'
            self.images = [img, ]

        # Params
        self.spline = spline
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.periodic = periodic
        self.show_fit = show_fit
        self.k = k

        # Internal
        self._current_frame = self.start_frame
        self._current_image = 0
        self._point0 = None
        self._points = None
        self._line = None
        self._fitted = False

        # Specify vlim
        if self.img_type == 'stack' or self.img_type == 'single':
            self.vmax = max([np.percentile(i, 99.9) for i in self.images])
            self.vmin = min([np.percentile(i, 0.1) for i in self.images])
        elif self.img_type == 'list':
            self.vmax = [np.percentile(i, 99.9) for i in self.images]
            self.vmin = [np.percentile(i, 0.1) for i in self.images]

        # Outputs
        self.xpoints = []
        self.ypoints = []
        self.roi = None

    def run(self):
        # Set up figure
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)

        # Stack
        if self.img_type == 'stack' or self.img_type == 'list':
            plt.subplots_adjust(left=0.25, bottom=0.25)
            self.axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
            if self.end_frame is None:
                self.end_frame = len(self.images)
            self.sframe = Slider(self.axframe, 'Frame', self.start_frame, self.end_frame, valinit=self.start_frame,
                                 valfmt='%d')
            self.sframe.on_changed(self.draw_frame)

        self.draw_frame(self.start_frame)

        # Show figure
        self.fig.canvas.set_window_title('Specify ROI')
        self.fig.canvas.mpl_connect('close_event', lambda event: self.fig.canvas.stop_event_loop())
        self.fig.canvas.start_event_loop(timeout=-1)

    def draw_frame(self, i: int):
        self._current_frame = i
        self.ax.clear()

        # Plot image
        if self.img_type == 'stack' or self.img_type == 'single':
            self.ax.imshow(self.images[int(i)], cmap='gray', vmin=self.vmin, vmax=self.vmax)
        else:
            self.ax.imshow(self.images[int(i)], cmap='gray', vmin=self.vmin[int(i)], vmax=self.vmax[int(i)])

        # Finalise figure
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.03, 0.97,
                     'Specify ROI clockwise (4 points minimum)'
                     '\nClick to lay points'
                     '\nBACKSPACE: undo'
                     '\nENTER: Save and continue',
                     color='white',
                     transform=self.ax.transAxes, fontsize=8, va='top', ha='left')
        self.display_points()
        self.fig.canvas.draw()

    def button_press_callback(self, event: MouseEvent):
        if not self._fitted:
            if isinstance(event.inaxes, type(self.ax)):
                # Add points to list
                self.xpoints.extend([event.xdata])
                self.ypoints.extend([event.ydata])

                # Display points
                self.display_points()
                self.fig.canvas.draw()

    def key_press_callback(self, event: KeyEvent):
        if event.key == 'backspace':
            if not self._fitted:
                # Remove last drawn point
                if len(self.xpoints) != 0:
                    self.xpoints = self.xpoints[:-1]
                    self.ypoints = self.ypoints[:-1]
                self.display_points()
                self.fig.canvas.draw()
            else:
                # Remove line
                self._fitted = False
                self._line.pop(0).remove()
                self.roi = None
                self.fig.canvas.draw()

        if event.key == 'enter':
            if len(self.xpoints) != 0:
                roi = np.vstack((self.xpoints, self.ypoints)).T

                # Spline
                if self.spline:
                    if not self._fitted:
                        self.roi = spline_roi(roi, periodic=self.periodic, k=self.k)
                        self._fitted = True

                        # Display line
                        if self.show_fit:
                            self._line = self.ax.plot(self.roi[:, 0], self.roi[:, 1], c='b')
                            self.fig.canvas.draw()
                        else:
                            plt.close(self.fig)
                    else:
                        plt.close(self.fig)
                else:
                    self.roi = roi
                    plt.close(self.fig)
            else:
                self.roi = []
                plt.close(self.fig)

    def display_points(self):
        # Remove existing points
        try:
            self._point0.remove()
            self._points.remove()
        except (ValueError, AttributeError) as error:
            pass

        # Plot all points
        if len(self.xpoints) != 0:
            self._points = self.ax.scatter(self.xpoints, self.ypoints, c='lime', s=10)
            self._point0 = self.ax.scatter(self.xpoints[0], self.ypoints[0], c='r', s=10)


class ROI_jupyter:

    def __init__(self,
                 img: Union[np.ndarray, list],
                 spline: bool = True,
                 start_frame: int = 0,
                 end_frame: Optional[int] = None,
                 periodic: bool = True,
                 show_fit: bool = True,
                 k: int = 3):

        self.fig = None
        self.ax = None

        # Detect if single frame or stack
        if type(img) is list:
            self.img_type = 'list'
            self.images = img

        elif len(img.shape) == 3:
            self.img_type = 'stack'
            self.images = list(img)
        else:
            self.img_type = 'single'
            self.images = [img, ]

        # Params
        self.spline = spline
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.periodic = periodic
        self.show_fit = show_fit
        self.k = k

        # Specify vlim
        self.vmax = max([np.percentile(i, 99.9) for i in self.images])
        self.vmin = min([np.percentile(i, 0.1) for i in self.images])

        # Outputs
        self.xpoints = []
        self.ypoints = []
        self.roi = None

    def run(self):
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)

        # Buttons
        self.ax_undo = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.b_undo = Button(self.ax_undo, 'Undo')
        self.b_undo.on_clicked(self._undo)
        self.ax_save = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.b_save = Button(self.ax_save, 'Save')
        self.b_save.on_clicked(self._save)

        # Stack
        if self.img_type == 'stack' or self.img_type == 'list':
            @widgets.interact(Frame=(0, len(self.images) - 1, 1))
            def update(Frame=0):
                self.draw_frame(Frame)
        else:
            self.draw_frame(0)

        self.fig.set_size_inches(4, 4)

    def button_press_callback(self, event: MouseEvent):
        if isinstance(event.inaxes, type(self.ax)):
            # Add points to list
            self.xpoints.extend([event.xdata])
            self.ypoints.extend([event.ydata])

            # Display points
            self.display_points()
            self.fig.canvas.draw()

    def _undo(self, _):
        # Remove last drawn point
        if len(self.xpoints) != 0:
            self.xpoints = self.xpoints[:-1]
            self.ypoints = self.ypoints[:-1]
        self.display_points()
        self.fig.canvas.draw()

    def _save(self, _):
        roi = np.vstack((self.xpoints, self.ypoints)).T

        # Spline
        if self.spline:
            self.roi = spline_roi(roi, periodic=self.periodic, k=self.k)

            # Display line
            if self.show_fit:
                self._line = self.ax.plot(self.roi[:, 0], self.roi[:, 1], c='b')
                self.fig.canvas.draw()
        else:
            self.roi = roi

        plt.close()

    def display_points(self):
        # Remove existing points
        try:
            self._point0.remove()
            self._points.remove()
        except (ValueError, AttributeError) as error:
            pass

        # Plot all points
        if len(self.xpoints) != 0:
            self._points = self.ax.scatter(self.xpoints, self.ypoints, c='lime', s=10)
            self._point0 = self.ax.scatter(self.xpoints[0], self.ypoints[0], c='r', s=10)

    def draw_frame(self, i: int):
        self.ax.clear()

        # Plot image
        self.ax.imshow(self.images[int(i)], cmap='gray', vmin=self.vmin, vmax=self.vmax)

        # Finalise figure
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.03, 0.97,
                     'Specify ROI clockwise from posterior (4 points minimum)'
                     '\nClick to lay points',
                     color='white',
                     transform=self.ax.transAxes, fontsize=8, va='top', ha='left')
        self.display_points()
        self.fig.canvas.draw()


def spline_roi(roi: np.ndarray, periodic: bool = True, s: float = 0.0, k: int = 3) -> np.ndarray:
    """
    Fits a spline to points specifying the coordinates of the cortex, then interpolates to pixel distances

    Args:
        roi:
        periodic:
        s:
        k:

    Returns:

    """

    # Append the starting x,y coordinates
    if periodic:
        x = np.r_[roi[:, 0], roi[0, 0]]
        y = np.r_[roi[:, 1], roi[0, 1]]
    else:
        x = roi[:, 0]
        y = roi[:, 1]

    # Fit spline
    tck, u = splprep([x, y], s=s, per=periodic, k=k)

    # Evaluate spline
    xi, yi = splev(np.linspace(0, 1, 10000), tck)

    # Interpolate
    return interp_roi(np.vstack((xi, yi)).T, periodic=periodic)


def interp_roi(roi: np.ndarray, periodic: bool = True, npoints: Optional[int] = None, gap: int = 1) -> np.ndarray:
    """
    Interpolates coordinates to one pixel distances (or as close as possible to one pixel)
    Linear interpolation

    Args:
        roi:
        periodic:
        npoints:
        gap:

    Returns:

    """

    if periodic:
        c = np.append(roi, [roi[0, :]], axis=0)
    else:
        c = roi

    # Calculate distance between points in pixel units
    distances = ((np.diff(c[:, 0]) ** 2) + (np.diff(c[:, 1]) ** 2)) ** 0.5
    distances_cumsum = np.r_[0, np.cumsum(distances)]
    total_length = sum(distances)

    # Interpolate
    fx, fy = interp1d(distances_cumsum, c[:, 0], kind='linear'), interp1d(distances_cumsum, c[:, 1], kind='linear')
    if npoints is None:
        positions = np.linspace(0, total_length, int(round(total_length / gap)))
    else:
        positions = np.linspace(0, total_length, npoints + 1)
    xcoors, ycoors = fx(positions), fy(positions)
    newpoints = np.c_[xcoors[:-1], ycoors[:-1]]
    return newpoints


def offset_coordinates(roi: np.ndarray, offsets: Union[np.ndarray, float], periodic: bool = True) -> np.ndarray:
    """
    Reads in coordinates, adjusts according to offsets

    Args:
        roi:  two column array containing x and y coordinates. e.g. coors = np.loadtxt(filename)
        offsets: array the same length as coors. Direction?
        periodic:

    Returns:
         array in same format as coors containing new coordinates.
         To save this in a fiji readable format:
         np.savetxt(filename, newcoors, fmt='%.4f', delimiter='\t')

    """

    # Calculate gradients
    xcoors = roi[:, 0]
    ycoors = roi[:, 1]
    if periodic:
        ydiffs = np.diff(ycoors, prepend=ycoors[-1])
        xdiffs = np.diff(xcoors, prepend=xcoors[-1])
    else:
        ydiffs = np.diff(ycoors)
        xdiffs = np.diff(xcoors)
        ydiffs = np.r_[ydiffs[0], ydiffs]
        xdiffs = np.r_[xdiffs[0], xdiffs]

    grad = ydiffs / xdiffs
    tangent_grad = -1 / grad

    # Offset coordinates
    xchange = ((offsets ** 2) / (1 + tangent_grad ** 2)) ** 0.5
    ychange = xchange / abs(grad)
    newxs = xcoors + np.sign(ydiffs) * np.sign(offsets) * xchange
    newys = ycoors - np.sign(xdiffs) * np.sign(offsets) * ychange
    newcoors = np.swapaxes(np.vstack([newxs, newys]), 0, 1)
    return newcoors
