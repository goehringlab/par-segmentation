from par_segmentation import *



class TestGdCompletion:
    """

    GRADIENT DESCENT COMPLETION TESTS

    Testing that the gradient descent optimiser runs to completion
    NOT testing that any results are as expected

    """

    path = os.path.dirname(os.path.abspath(__file__)) + '/../scripts'
    imgs = [load_image(os.path.dirname(os.path.abspath(__file__)) + '/../scripts/nwg338_af_corrected.tif'),]
    rois = [np.loadtxt(os.path.dirname(os.path.abspath(__file__)) + '/../scripts/nwg338_ROI_manual.txt'),]

    def test_1(self):
        # Testing that it runs to completion with default parameters
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False)
        iq.run()
        res = iq.compile_res()

    def test_2(self):
        # Testing that it runs to completion with periodic False
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, periodic=False)
        iq.run()

    def test_3(self):
        # Testing that it runs to completion with rotate True
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, rotate=True)
        iq.run()

    def test_4(self):
        # Testing that it runs to completion with adaptive sigma True
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, adaptive_sigma=True)
        iq.run()

    def test_5(self):
        # Testing that it runs to completion with adaptive batch_norm True
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, batch_norm=True)
        iq.run()

    def test_6(self):
        # Testing that it runs to completion with fit_outer False
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, fit_outer=False)
        iq.run()

    def test_7(self):
        # Testing that it runs to completion with nfits None
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, nfits=None)
        iq.run()

    def test_8(self):
        # Testing that it runs to completion with zerocap True
        iq = ImageQuant(img=self.imgs, roi=self.rois, method='GD', descent_steps=10, verbose=False, zerocap=True)
        iq.run()
