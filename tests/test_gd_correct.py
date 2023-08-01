from par_segmentation import *
import pytest


class TestGdCorrect:
    """
    Making sure results from gradient descent optimiser are as expected
    Note: if underlying algorithm is changed, or default parameters are changed, tests may fail

    """

    path = os.path.dirname(os.path.abspath(__file__)) + '/../scripts'
    imgs = [load_image(os.path.dirname(os.path.abspath(__file__)) + '/../scripts/nwg338_af_corrected.tif'),]
    rois = [np.loadtxt(os.path.dirname(os.path.abspath(__file__)) + '/../scripts/nwg338_ROI_manual.txt'),]

    def test_1(self):
        # Correct results when quantifying the image
        iq = ImageQuant(img=self.imgs[0], roi=self.rois[0], method='GD', verbose=False)
        iq.run()
        res = iq.compile_res()
        
        assert res.iloc[0]['Frame'] == 0
        assert res.iloc[0]['Position'] == 0
        assert res.iloc[0]['Membrane signal'] == pytest.approx(6924.348109365306)
        assert res.iloc[0]['Cytoplasmic signal'] == pytest.approx(6995.061025591719)
        assert iq.roi[0][0, 0] == pytest.approx(182.18897189832285)
