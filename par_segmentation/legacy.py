import numpy as np
import cv2
from .funcs import offset_coordinates


def polycrop(img: np.ndarray, polyline: np.ndarray, enlarge: float) -> np.ndarray:
    """
    Crops image according to polyline coordinates by setting values not contained within the coordinates to zero

    Args:
        img: numpy array of image
        polyline: roi specifying the bounding region (two columns specifying x and y coordinates)
        enlarge: amount by which to expand or contract the ROI (pixel units)

    Returns:
        numpy array of same shape img, with regions outside of polyline set to zero

    """

    newcoors = np.int32(
        offset_coordinates(polyline, enlarge * np.ones([len(polyline[:, 0])]))
    )
    mask = np.zeros(img.shape)
    mask = cv2.fillPoly(mask, [newcoors], 1)
    newimg = img * mask
    return newimg


def bg_subtraction(
    img: np.ndarray, roi: np.ndarray, band: tuple = (25, 75)
) -> np.ndarray:
    """

    Subtracts background intensity from an image of a cell. Background intensity calculated as the mean intensity within
    a band surronding the cell (specified by ROI)

    Args:
        img: numpy array of image to subtract background from
        roi: two column numpy array specifying coordinates of the cell boundary
        band: inner and outer distance of the band from the roi

    Returns:
        numpy array of image with background subtracted

    """
    a = polycrop(img, roi, band[1]) - polycrop(img, roi, band[0])
    a = [np.nanmean(a[np.nonzero(a)])]
    return img - a


def calc_vol(normcoors: np.ndarray) -> float:
    r1 = (max(normcoors[:, 0]) - min(normcoors[:, 0])) / 2
    r2 = (max(normcoors[:, 1]) - min(normcoors[:, 1])) / 2
    return (4 / 3) * np.pi * r2 * r2 * r1


def calc_sa(normcoors: np.ndarray) -> float:
    r1 = (max(normcoors[:, 0]) - min(normcoors[:, 0])) / 2
    r2 = (max(normcoors[:, 1]) - min(normcoors[:, 1])) / 2
    e = (1 - (r2**2) / (r1**2)) ** 0.5
    return 2 * np.pi * r2 * r2 * (1 + (r1 / (r2 * e)) * np.arcsin(e))
