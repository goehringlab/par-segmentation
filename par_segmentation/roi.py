import numpy as np
from scipy.interpolate import interp1d, splev, splprep

"""
Todo: This no longer works with multiple channels - intensity ranges
Todo: Ability to specify a directory and open all channels. Or an nd file

"""


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
    x, y = roi[:, 0], roi[:, 1]

    # Append the starting x,y coordinates if the ROI is periodic
    if periodic:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

    # Fit spline and evaluate it
    tck, _ = splprep([x, y], s=s, per=periodic, k=k)
    xi, yi = splev(np.linspace(0, 1, 10000), tck)

    # Interpolate
    return interp_roi(np.vstack((xi, yi)).T, periodic=periodic)


def interp_roi(
    roi: np.ndarray, periodic: bool = True, npoints: int | None = None, gap: int = 1
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
    c = np.append(roi, [roi[0, :]], axis=0) if periodic else roi

    # Calculate distance between points in pixel units
    distances = np.sqrt(np.sum(np.diff(c, axis=0) ** 2, axis=1))
    distances_cumsum = np.r_[0, np.cumsum(distances)]
    total_length = distances_cumsum[-1]

    # Interpolate
    fx, fy = interp1d(distances_cumsum, c[:, 0]), interp1d(distances_cumsum, c[:, 1])
    positions = np.linspace(
        0, total_length, npoints + 1 if npoints else int(round(total_length / gap))
    )
    newpoints = np.column_stack((fx(positions), fy(positions)))[:-1]
    return newpoints


def offset_coordinates(
    roi: np.ndarray, offsets: np.ndarray | float, periodic: bool = True
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
    xchange = np.sqrt((offsets**2) / (1 + tangent_grad**2))
    ychange = xchange / np.abs(grad)
    newxs = xcoors + np.sign(ydiffs) * np.sign(offsets) * xchange
    newys = ycoors - np.sign(xdiffs) * np.sign(offsets) * ychange
    return np.column_stack([newxs, newys])
