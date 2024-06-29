import numpy as np
from par_segmentation import make_mask, load_image
from par_segmentation.legacy import polycrop
import os


def test_make_mask():
    img = load_image(
        os.path.dirname(os.path.abspath(__file__))
        + "/../scripts/nwg338_af_corrected.tif"
    )
    roi = np.loadtxt(
        os.path.dirname(os.path.abspath(__file__)) + "/../scripts/nwg338_ROI_manual.txt"
    )
    result = make_mask(img.shape, roi)
    assert np.sum(~np.isnan(result)) == 17033
    assert np.sum(~np.isnan(result[300, :])) == 178


def test_polycrop():
    img = load_image(
        os.path.dirname(os.path.abspath(__file__))
        + "/../scripts/nwg338_af_corrected.tif"
    )
    roi = np.loadtxt(
        os.path.dirname(os.path.abspath(__file__)) + "/../scripts/nwg338_ROI_manual.txt"
    )
    result = polycrop(img, roi, 0)
    assert int(result.sum()) == 105625425
    assert np.count_nonzero(result) == 17033
    assert np.count_nonzero(result[300, :]) == 178
