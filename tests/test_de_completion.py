import unittest
from par_segmentation import *


class DeCompletionTests(unittest.TestCase):
    """
    Testing that the differential evolution optimiser runs to completion
    NOT testing that any results are as expected

    """

    @classmethod
    def setUpClass(cls):
        path = os.path.dirname(os.path.abspath(__file__)) + '/../data/dataset2_par2_neon'
        paths = direcslist(path)[:2]
        cls.imgs = [load_image(p + '/af_corrected.tif') for p in paths]
        cls.rois = [np.loadtxt(p + '/ROI.txt') for p in paths]

    # def test1(self):
    #     # Testing that it runs to completion with default parameters
    #     iq = ImageQuant(img=self.imgs, roi=self.rois, method='DE', verbose=False, parallel=False)
    #     iq.run()
    #     res = iq.compile_res()
