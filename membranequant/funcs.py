import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import CubicSpline
from scipy.special import erf
from skimage import io
import cv2
import glob
import copy
import os
from .roi import offset_coordinates
from typing import Optional

"""

"""


########## IMAGE HANDLING ###########


def load_image(filename: str) -> np.ndarray:
    """
    Given the filename of a TIFF, creates numpy array with pixel intensities

    Args:
        filename:

    Returns:

    """

    return io.imread(filename).astype(float)


def save_img(img: np.ndarray, direc: str):
    """
    Saves 2D array as .tif file

    Args:
        img:
        direc:

    Returns:

    """

    io.imsave(direc, img.astype('float32'))


def save_img_jpeg(img: np.ndarray, direc: str, cmin: Optional[float] = None, cmax: Optional[float] = None,
                  cmap: str = 'gray'):
    """
    Saves 2D array as jpeg, according to min and max pixel intensities

    Args:
        img:
        direc:
        cmin:
        cmax:
        cmap:

    Returns:

    """

    plt.imsave(direc, img, vmin=cmin, vmax=cmax, cmap=cmap)


########### IMAGE OPERATIONS ###########


def straighten(img: np.ndarray, roi: np.ndarray, thickness: int, periodic: bool = True, interp: str = 'cubic',
               ninterp: Optional[int] = None) -> np.ndarray:
    """
    Creates straightened image based on coordinates
    Todo: Doesn't work properly for non-periodic rois

    Args:
        img:
        roi: Coordinates. Should be 1 pixel length apart in a loop
        thickness:
        periodic:
        interp:
        ninterp:

    Returns:

    """

    if ninterp is None:
        ninterp = thickness

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

    # Get interpolation coordinates
    offsets = np.linspace(thickness / 2, -thickness / 2, ninterp)
    xchange = ((offsets ** 2)[np.newaxis, :] / (1 + tangent_grad ** 2)[:, np.newaxis]) ** 0.5
    ychange = xchange / abs(grad)[:, np.newaxis]
    gridcoors_x = xcoors[:, np.newaxis] + np.sign(ydiffs)[:, np.newaxis] * np.sign(offsets)[np.newaxis, :] * xchange
    gridcoors_y = ycoors[:, np.newaxis] - np.sign(xdiffs)[:, np.newaxis] * np.sign(offsets)[np.newaxis, :] * ychange

    # Interpolate
    if interp == 'linear':
        straight = map_coordinates(img.T, [gridcoors_x, gridcoors_y], order=1, mode='nearest')
    elif interp == 'cubic':
        straight = map_coordinates(img.T, [gridcoors_x, gridcoors_y], order=3, mode='nearest')
    else:
        raise ValueError('interp must be "linear" or "cubic"')
    return straight.astype(np.float64).T


def polycrop(img: np.ndarray, polyline: np.ndarray, enlarge: float) -> np.ndarray:
    """
    Crops image according to polyline coordinates
    Expand or contract selection with enlarge parameter

    Args:
        img:
        polyline:
        enlarge:

    Returns:

    """

    newcoors = np.int32(offset_coordinates(polyline, enlarge * np.ones([len(polyline[:, 0])])))
    mask = np.zeros(img.shape)
    mask = cv2.fillPoly(mask, [newcoors], 1)
    newimg = img * mask
    return newimg


def rotated_embryo(img: np.ndarray, roi: np.ndarray, l: int, h: int, order: int = 1,
                   return_roi: bool = False) -> np.ndarray:
    """
    Takes an image and rotates according to coordinates so that anterior is on left, posterior on right
    PROBLEM: some of the returned coordinates are anticlockwise

    Args:
        img:
        roi:
        l:
        h:
        order:
        return_roi:

    Returns:

    """

    # PCA on ROI coordinates
    [_, coeff] = np.linalg.eig(np.cov(roi.T))

    # Transform ROI
    roi_transformed = np.dot(coeff.T, roi.T)

    # Force long axis orientation
    x_range = (min(roi_transformed[0, :]) - max(roi_transformed[0, :]))
    y_range = (min(roi_transformed[1, :]) - max(roi_transformed[1, :]))
    if x_range > y_range:
        img = img.T
        roi_transformed = np.flipud(roi_transformed)
        coeff = coeff.T

    # Coordinate grid
    centre_x = (min(roi_transformed[0, :]) + max(roi_transformed[0, :])) / 2
    xvals = np.arange(int(centre_x) - (l / 2), int(centre_x) + (l / 2))
    centre_y = (min(roi_transformed[1, :]) + max(roi_transformed[1, :])) // 2
    yvals = np.arange(int(centre_y) - (h / 2), int(centre_y) + (h / 2))
    xvals_grid = np.tile(xvals, [len(yvals), 1])
    yvals_grid = np.tile(yvals, [len(xvals), 1]).T
    roi_transformed = roi_transformed - np.expand_dims([centre_x - (l / 2), centre_y - (h / 2)], -1)

    # Transform coordinate grid back
    [xvals_back, yvals_back] = np.dot(coeff, np.ndarray([xvals_grid.flatten(), yvals_grid.flatten()]))
    xvals_back_grid = np.reshape(xvals_back, [len(yvals), len(xvals)])
    yvals_back_grid = np.reshape(yvals_back, [len(yvals), len(xvals)])

    # Map coordinates using linear interpolation
    zvals = map_coordinates(img.T, [xvals_back_grid, yvals_back_grid], order=order)

    # Force posterior on right
    if roi_transformed[0, 0] < roi_transformed[0, roi_transformed.shape[1] // 2]:
        zvals = np.fliplr(zvals)
        roi_transformed[0, :] = l - roi_transformed[0, :]

    if return_roi:
        return zvals, roi_transformed.T
    else:
        return zvals


def bg_subtraction(img: np.ndarray, roi: np.ndarray, band: tuple = (25, 75)) -> np.ndarray:
    """

    Args:
        img:
        roi:
        band:

    Returns:

    """
    a = polycrop(img, roi, band[1]) - polycrop(img, roi, band[0])
    a = [np.nanmean(a[np.nonzero(a)])]
    return img - a


########### ROI OPERATIONS ###########


def rotate_roi(roi: np.ndarray) -> np.ndarray:
    """
    Rotates coordinate array so that most posterior point is at the beginning

    Args:
        roi:

    Returns:

    """

    # PCA to find long axis
    M = (roi - np.mean(roi.T, axis=1)).T
    [_, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M)

    # Find most extreme points
    a = np.argmin(np.minimum(score[0, :], score[1, :]))
    b = np.argmax(np.maximum(score[0, :], score[1, :]))

    # Find the one closest to user defined posterior
    dista = np.hypot((roi[0, 0] - roi[a, 0]), (roi[0, 1] - roi[a, 1]))
    distb = np.hypot((roi[0, 0] - roi[b, 0]), (roi[0, 1] - roi[b, 1]))

    # Rotate coordinates
    if dista < distb:
        newcoors = np.roll(roi, len(roi[:, 0]) - a, 0)
    else:
        newcoors = np.roll(roi, len(roi[:, 0]) - b, 0)

    return newcoors


def norm_roi(roi: np.ndarray):
    """
    Aligns coordinates to their long axis

    Args:
        roi:

    Returns:

    """

    # PCA
    M = (roi - np.mean(roi.T, axis=1)).T
    [_, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M).T

    # Find long axis
    if (max(score[0, :]) - min(score[0, :])) < (max(score[1, :]) - min(score[1, :])):
        score = np.fliplr(score)

    return score


########### ARRAY OPERATIONS ###########


def interp_1d_array(array: np.ndarray, n: int, method: str = 'cubic') -> np.ndarray:
    """
    Interpolates a one dimensional array into n points

    TODO: Combine with 2d function

    Args:
        array:
        n:
        method:

    Returns:

    """

    if method == 'linear':
        return np.interp(np.linspace(0, len(array) - 1, n), np.array(range(len(array))), array)
    elif method == 'cubic':
        return CubicSpline(np.arange(len(array)), array)(np.linspace(0, len(array) - 1, n))


def interp_2d_array(array: np.ndarray, n: int, ax: int = 0, method: str = 'cubic') -> np.ndarray:
    """
    Interpolates values along y axis into n points, for each x value
    Todo: no loops

    Args:
        array:
        n:
        ax:
        method:

    Returns:

    """

    if ax == 0:
        interped = np.zeros([n, len(array[0, :])])
        for x in range(len(array[0, :])):
            interped[:, x] = interp_1d_array(array[:, x], n, method)
        return interped
    elif ax == 1:
        interped = np.zeros([len(array[:, 0]), n])
        for x in range(len(array[:, 0])):
            interped[x, :] = interp_1d_array(array[x, :], n, method)
        return interped
    else:
        raise ValueError('ax must be 0 or 1')


def rolling_ave_1d(array: np.ndarray, window: int, periodic: bool = True) -> np.ndarray:
    """

    Args:
        array:
        window:
        periodic:

    Returns:

    """

    if window == 1:
        return array
    if not periodic:
        array_padded = np.r_[array[:int(window / 2)][::-1], array, array[-int(window / 2):][::-1]]
    else:
        array_padded = np.r_[array[-int(window / 2):], array, array[:int(window / 2)]]
    cumsum = np.cumsum(array_padded)
    return (cumsum[window:] - cumsum[:-window]) / window


def rolling_ave_2d(array: np.ndarray, window: int, periodic: bool = True) -> np.ndarray:
    """
    Returns rolling average across the x axis of an image (used for straightened profiles)

    Args:
        array: image data
        window: number of pixels to average over. Odd number is best
        periodic: if true, rolls over at ends

    Returns:

    """

    if window == 1:
        return array
    if not periodic:
        array_padded = np.c_[array[:, :int(window / 2)][:, :-1], array, array[:, -int(window / 2):][:, :-1]]
    else:
        array_padded = np.c_[array[:, -int(window / 2):], array, array[:, :int(window / 2)]]
    cumsum = np.cumsum(array_padded, axis=1)
    return (cumsum[:, window:] - cumsum[:, :-window]) / window


def bounded_mean_1d(array: np.ndarray, bounds: tuple, weights: Optional[np.ndarray] = None) -> float:
    """
    Averages 1D array over region specified by bounds
    Array and weights should be same length
    Todo: Should add interpolation step first?

    Args:
        array:
        bounds:
        weights:

    Returns:

    """

    if weights is None:
        weights = np.ones([len(array)])
    if bounds[0] < bounds[1]:
        mean = np.average(array[int(len(array) * bounds[0]): int(len(array) * bounds[1] + 1)],
                          weights=weights[int(len(array) * bounds[0]): int(len(array) * bounds[1] + 1)])
    else:
        mean = np.average(np.hstack((array[:int(len(array) * bounds[1] + 1)], array[int(len(array) * bounds[0]):])),
                          weights=np.hstack(
                              (weights[:int(len(array) * bounds[1] + 1)], weights[int(len(array) * bounds[0]):])))
    return mean


def bounded_mean_2d(array: np.ndarray, bounds: tuple) -> np.ndarray:
    """
    Averages 2D array in y dimension over region specified by bounds
    Todo: Should add axis parameter
    Todo: Should add interpolation step first

    Args:
        array:
        bounds:

    Returns:

    """

    if bounds[0] < bounds[1]:
        mean = np.mean(array[:, int(len(array[0, :]) * bounds[0]): int(len(array[0, :]) * bounds[1])], 1)
    else:
        mean = np.mean(
            np.hstack((array[:, :int(len(array[0, :]) * bounds[1])], array[:, int(len(array[0, :]) * bounds[0]):])), 1)
    return mean


########### REFERENCE PROFILES ###########


def gaus(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    """
    Create Gaussian curve with centre and width specified

    Args:
        x:
        centre:
        width:

    Returns:

    """

    return np.exp(-((x - centre) ** 2) / (2 * width ** 2))


def error_func(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    """
    Create error function with centre and width specified

    Args:
        x:
        centre:
        width:

    Returns:

    """

    return erf((x - centre) / width)


########### MISC FUNCTIONS ###########


def asi(mems: np.ndarray, size: float) -> float:
    """
    Calculates asymmetry index based on membrane concentration profile

    """

    ant = bounded_mean_1d(mems, (0.5 - size / 2, 0.5 + size / 2))
    post = bounded_mean_1d(mems, (1 - size / 2, size / 2))
    return (ant - post) / (2 * (ant + post))


def dosage(img: np.ndarray, roi: np.ndarray, expand: float) -> np.ndarray:
    return np.nanmean(img * make_mask((512, 512), offset_coordinates(roi, expand)))


def calc_vol(normcoors: np.ndarray) -> float:
    r1 = (max(normcoors[:, 0]) - min(normcoors[:, 0])) / 2
    r2 = (max(normcoors[:, 1]) - min(normcoors[:, 1])) / 2
    return (4 / 3) * np.pi * r2 * r2 * r1


def calc_sa(normcoors: np.ndarray) -> float:
    r1 = (max(normcoors[:, 0]) - min(normcoors[:, 0])) / 2
    r2 = (max(normcoors[:, 1]) - min(normcoors[:, 1])) / 2
    e = (1 - (r2 ** 2) / (r1 ** 2)) ** 0.5
    return 2 * np.pi * r2 * r2 * (1 + (r1 / (r2 * e)) * np.arcsin(e))


def make_mask(shape: tuple, roi: np.ndarray) -> np.ndarray:
    return cv2.fillPoly(np.zeros(shape) * np.nan, [np.int32(roi)], 1)


def readnd(path: str) -> dict:
    """

    Args:
        path: directory to embryo folder containing nd file

    Returns:
        dictionary containing data from nd file

    """

    nd = {}
    f = open(path, 'r').readlines()
    for line in f[:-1]:
        nd[line.split(', ')[0].replace('"', '')] = line.split(', ')[1].strip().replace('"', '')
    return nd


def organise_by_nd(path: str):
    """
    Organises images in a folder using the nd files

    Args:
        path:

    Returns:

    """
    a = glob.glob(f'{path}/*.nd')
    for b in a:
        name = os.path.basename(os.path.normpath(b))
        if name[0] == '_':
            folder = name[1:-3]
        else:
            folder = name[:-3]
        os.makedirs(f'{path}/{folder}')
        os.rename(b, f'{path}/{folder}/{name}')
        for file in glob.glob(f'{b[:-3]}_*'):
            os.rename(file, f'{path}/{folder}/{os.path.basename(os.path.normpath(file))}')


def _direcslist(dest: str, levels: int = 0, exclude: Optional[tuple] = ('!',),
                exclusive: Optional[tuple] = None) -> list:
    lis = sorted(glob.glob(f'{dest}/*/'))

    for level in range(levels):
        newlis = []
        for e in lis:
            newlis.extend(sorted(glob.glob(f'{e}/*/')))
        lis = newlis
        lis = [x[:-1] for x in lis]

    # Excluded directories
    lis_copy = copy.deepcopy(lis)
    if exclude is not None:
        for x in lis:
            for i in exclude:
                if i in x:
                    lis_copy.remove(x)
                    break

    # Exclusive directories
    if exclusive is not None:
        lis2 = []
        for x in lis_copy:
            for i in exclusive:
                if i in x:
                    lis2.append(x)
    else:
        lis2 = lis_copy

    return sorted(lis2)


def direcslist(dest: str, levels: int = 0, exclude: Optional[tuple] = ('!',),
               exclusive: Optional[tuple] = None) -> list:
    """
    Gives a list of directories in a given directory (full path)
    Todo: os.walk

    Args:
        dest:
        levels:
        exclude: exclude directories containing this string
        exclusive: exclude directories that don't contain this string

    Returns:

    """

    if type(dest) is list:
        out = []
        for d in dest:
            out.extend(_direcslist(d, levels, exclude, exclusive))
        return out
    else:
        return _direcslist(dest, levels, exclude, exclusive)


def in_notebook():
    """
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

    """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
