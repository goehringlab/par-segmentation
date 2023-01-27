import unittest
from par_segmentation import *


class DeCorrectTests(unittest.TestCase):
    """
    Making sure results from differential evolution optimiser are as expected
    Note: if underlying algorithm is changed, or default parameters are changed, tests may fail

    """

    @classmethod
    def setUpClass(cls):
        path = os.path.dirname(os.path.abspath(__file__)) + '/../data/dataset2_par2_neon'
        paths = direcslist(path)[:2]
        cls.imgs = [load_image(p + '/af_corrected.tif') for p in paths]
        cls.rois = [np.loadtxt(p + '/ROI.txt') for p in paths]

    # def test1(self):
    #     # Correct results when quantifying image 1
    #     np.random.seed(12345)  # <- as it uses a stochastic algorithm
    #     iq = ImageQuant(img=self.imgs[0], roi=self.rois[0], method='DE', verbose=False)
    #     iq.run()
    #     res = iq.compile_res()
    #
    #     self.assertEqual(res.iloc[0]['Frame'], 0)
    #     self.assertEqual(res.iloc[0]['Position'], 0)
    #     self.assertAlmostEqual(res.iloc[0]['Membrane signal'], 25636.51045019628)
    #     self.assertAlmostEqual(res.iloc[0]['Cytoplasmic signal'], 9383.178431357195)
    #     self.assertAlmostEqual(iq.roi[0][0, 0], 286.5988590949836)
