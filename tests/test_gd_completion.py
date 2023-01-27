import unittest
from par_segmentation import *


class GdCompletionTests(unittest.TestCase):
    """

    GRADIENT DESCENT COMPLETION TESTS

    Testing that the gradient descent optimiser runs to completion
    NOT testing that any results are as expected

    """

    @classmethod
    def setUpClass(cls):
        path = os.path.dirname(os.path.abspath(__file__)) + '/../data/dataset2_par2_neon'
        paths = direcslist(path)[:2]
        cls.imgs = [load_image(p + '/af_corrected.tif') for p in paths]
        cls.rois = [np.loadtxt(p + '/ROI.txt') for p in paths]

    def test1(self):
        # Testing that it runs to completion with default parameters
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False)
        iq.run()
        res = iq.compile_res()

    def test2(self):
        # Testing that it runs to completion with periodic False
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, periodic=False)
        iq.run()

    def test3(self):
        # Testing that it runs to completion with rotate True
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, rotate=True)
        iq.run()

    def test4(self):
        # Testing that it runs to completion with adaptive sigma True
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, adaptive_sigma=True)
        iq.run()

    def test5(self):
        # Testing that it runs to completion with adaptive batch_norm True
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, batch_norm=True)
        iq.run()

    def test6(self):
        # Testing that it runs to completion with fit_outer False
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, fit_outer=False)
        iq.run()

    def test7(self):
        # Testing that it runs to completion with nfits None
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, nfits=None)
        iq.run()

    def test8(self):
        # Testing that it runs to completion with zerocap True
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, zerocap=True)
        iq.run()
