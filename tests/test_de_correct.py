import os

import numpy as np
import pytest

from par_segmentation import load_image
from par_segmentation.quantifier import ImageQuant


class TestDeCorrectTests:
    """
    Making sure results from differential evolution optimiser are as expected
    Note: if underlying algorithm is changed, or default parameters are changed, tests may fail

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

    def test_1(self):
        # Correct results when quantifying the image
        np.random.seed(12345)  # <- as it uses a stochastic algorithm
        iq = ImageQuant(img=self.imgs[0], roi=self.rois[0], method="DE", verbose=False)
        iq.run()
        res = iq.compile_res()

        assert res.iloc[0]["Frame"] == 0
        assert res.iloc[0]["Position"] == 0
        assert res.iloc[0]["Membrane signal"] == pytest.approx(
            7222.288644674427, rel=1e-4
        )
        assert res.iloc[0]["Cytoplasmic signal"] == pytest.approx(
            7186.560758626298, rel=1e-4
        )
        assert iq.roi[0][0, 0] == pytest.approx(181.5831198442061, rel=1e-4)
