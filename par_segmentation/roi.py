from typing import Optional, Union

import numpy as np
from scipy.interpolate import interp1d, splev, splprep

"""
Todo: This no longer works with multiple channels - intensity ranges
Todo: Ability to specify a directory and open all channels. Or an nd file

"""

__all__ = ["spline_roi", "interp_roi", "offset_coordinates"]


def spline_roi(
    roi: np.ndarray, periodic: bool = True, s: float = 0.0, k: int = 3
) -> np.ndarray:
    """
    Fits a spline to points specifying the coordinates of the cortex, then interpolates to pixel distances

    Args:
        roi: two column array containing x and y coordinates. e.g. roi = np.loadtxt(filename)
        periodic: set to True if the ROI is periodic
        s: splprep s parameter
        k: splprep k parameter (spline order)

    Returns:
        spline ROI (numpy array)

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


def interp_roi(
    roi: np.ndarray, periodic: bool = True, npoints: Optional[int] = None, gap: int = 1
) -> np.ndarray:
    """
    Interpolates coordinates to one pixel distances (or as close as possible to one pixel). Linear interpolation

    Args:
        roi: two column array containing x and y coordinates. e.g. roi = np.loadtxt(filename)
        periodic: set to True if the ROI is periodic
        npoints: number of points to interpolate to
        gap: alternatively, specify the desired gap between succesive coordinates in pixel units

    Returns:
        interpolated ROI (numpy array)

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
    fx, fy = interp1d(distances_cumsum, c[:, 0], kind="linear"), interp1d(
        distances_cumsum, c[:, 1], kind="linear"
    )
    if npoints is None:
        positions = np.linspace(0, total_length, int(round(total_length / gap)))
    else:
        positions = np.linspace(0, total_length, npoints + 1)
    xcoors, ycoors = fx(positions), fy(positions)
    newpoints = np.c_[xcoors[:-1], ycoors[:-1]]
    return newpoints


def offset_coordinates(
    roi: np.ndarray, offsets: Union[np.ndarray, float], periodic: bool = True
) -> np.ndarray:
    """
    Reads in coordinates, adjusts according to offsets

    Args:
        roi:  two column array containing x and y coordinates. e.g. roi = np.loadtxt(filename)
        offsets: array the same length as coors. Direction?
        periodic: set to True if the ROI is periodic

    Returns:
         array in same format as coors containing new coordinates.\n
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
    xchange = ((offsets**2) / (1 + tangent_grad**2)) ** 0.5
    ychange = xchange / abs(grad)
    newxs = xcoors + np.sign(ydiffs) * np.sign(offsets) * xchange
    newys = ycoors - np.sign(xdiffs) * np.sign(offsets) * ychange
    newcoors = np.swapaxes(np.vstack([newxs, newys]), 0, 1)
    return newcoors
