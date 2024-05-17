import os

import numpy as np

from par_segmentation import load_image
from par_segmentation.quantifier import ImageQuant


class TestFlexiCompletion:
    """

    FLEXI MODEL COMPLETION TESTS

    Testing that the flexi model runs to completion
    NOT testing that any results are as expected

    """

    path = os.path.dirname(os.path.abspath(__file__)) + "/../scripts"
    imgs = [
        load_image(
            os.path.dirname(os.path.abspath(__file__))
            + "/../scripts/nwg338_af_corrected.tif"
        ),
    ]
    rois = [
        np.loadtxt(
            os.path.dirname(os.path.abspath(__file__))
            + "/../scripts/nwg338_ROI_manual.txt"
        ),
    ]

    def test_default(self):
        # Testing that it runs to completion with default parameters
        iq = ImageQuant(
            img=self.imgs,
            roi=self.rois,
            method="flexi",
        )
        iq.quantify()
        iq.compile_res()

    def test_nfits_none(self):
        # Testing that it runs to completion with nfits None
        iq = ImageQuant(
            img=self.imgs,
            roi=self.rois,
            method="flexi",
            nfits=None,
        )
        iq.quantify()

    def test_zerocap_false(self):
        # Testing that it runs to completion with zerocap False
        iq = ImageQuant(
            img=self.imgs,
            roi=self.rois,
            method="flexi",
            zerocap=True,
        )
        iq.quantify()

    def test_multiple_images(self):
        # Testing that it runs to completion with multiple images
        iq = ImageQuant(
            img=self.imgs + self.imgs,
            roi=self.rois + self.rois,
            method="flexi",
        )
        iq.quantify()
        iq.compile_res()
