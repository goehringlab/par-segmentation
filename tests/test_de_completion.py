from par_segmentation import *


class TestDeCompletion:
    """
    Testing that the differential evolution optimiser runs to completion
    NOT testing that any results are as expected

    """

    path = os.path.dirname(os.path.abspath(__file__)) + '/../scripts'
    imgs = [load_image(os.path.dirname(os.path.abspath(__file__)) + '/../scripts/nwg338_af_corrected.tif'),]
    rois = [np.loadtxt(os.path.dirname(os.path.abspath(__file__)) + '/../scripts/nwg338_ROI_manual.txt'),]

    def test_1(self):
        # Testing that it runs to completion with default parameters
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='DE', verbose=False, parallel=False)
        iq.run()
        res = iq.compile_res()
