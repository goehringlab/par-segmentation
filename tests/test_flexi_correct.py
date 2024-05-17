import os

import numpy as np
import pytest

from par_segmentation import load_image
from par_segmentation.quantifier import ImageQuant


class TestFlexiCorrect:
    """
    Making sure results from flexi model are as expected
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
        iq = ImageQuant(img=self.imgs[0], roi=self.rois[0], method="flexi")
        iq.quantify()
        res = iq.compile_res()

        assert res.iloc[0]["Frame"] == 0
        assert res.iloc[0]["Position"] == 0
        print(iq.roi[0][0, 0])
        assert res.iloc[0]["Membrane signal"] == pytest.approx(11081.1796875, rel=1e-4)
        assert res.iloc[0]["Cytoplasmic signal"] == pytest.approx(
            5736.6279296875, rel=1e-4
        )
        assert iq.roi[0][0, 0] == pytest.approx(181.7987012987013, rel=1e-4)
