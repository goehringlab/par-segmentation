import unittest
from discco import *

path = 'data/dataset2_par2_neon'
paths = direcslist(path)[:2]
imgs = [load_image(p + '/af_corrected.tif') for p in paths]
rois = [np.loadtxt(p + '/ROI.txt') for p in paths]


# Script not working as intended - often just runs one test instead of all of them
# TODO: write tests for plotting functions and autofluorescence functions

class GdCompletionTests(unittest.TestCase):
    """

    GRADIENT DESCENT COMPLETION TESTS

    Testing that the gradient descent optimiser runs to completion
    NOT testing that any results are as expected

    """

    def test1(self):
        # Testing that it runs to completion with default parameters
        iq = Discco(img=imgs, roi=rois, method='GD', descent_steps=10, verbose=False)
        iq.run()
        res = iq.compile_res()

    def test2(self):
        # Testing that it runs to completion with periodic False
        iq = Discco(img=imgs, roi=rois, method='GD', descent_steps=10, verbose=False, periodic=False)
        iq.run()

    def test3(self):
        # Testing that it runs to completion with rotate True
        iq = Discco(img=imgs, roi=rois, method='GD', descent_steps=10, verbose=False, rotate=True)
        iq.run()

    def test4(self):
        # Testing that it runs to completion with adaptive sigma True
        iq = Discco(img=imgs, roi=rois, method='GD', descent_steps=10, verbose=False, adaptive_sigma=True)
        iq.run()

    def test5(self):
        # Testing that it runs to completion with adaptive batch_norm True
        iq = Discco(img=imgs, roi=rois, method='GD', descent_steps=10, verbose=False, batch_norm=True)
        iq.run()

    def test6(self):
        # Testing that it runs to completion with fit_outer True
        iq = Discco(img=imgs, roi=rois, method='GD', descent_steps=10, verbose=False, fit_outer=True)
        iq.run()


class DeCompletionTests(unittest.TestCase):
    """
    Testing that the differential evolution optimiser runs to completion
    NOT testing that any results are as expected

    """

    def test1(self):
        # Testing that it runs to completion with default parameters
        iq = Discco(img=imgs, roi=rois, method='DE', verbose=False, parallel=False)
        iq.run()
        res = iq.compile_res()
        # Weirdly when this is uncommented the script will sometimes only run this test


# class GdCorrectTests(unittest.TestCase):
#     """
#     Making sure results from gradient descent optimiser are as expected
#     Note: if underlying algorithm is changed, or default parameters are changed, tests may fail
#
#     """
#
#     def test1(self):
#         # Correct results when quantifying image 1
#         iq = ImageQuant(img=imgs[0], roi=rois[0], method='GD', verbose=False)
#         iq.run()
#         res = iq.compile_res()
#         self.assertEqual(res.iloc[0]['Frame'], 0)
#         self.assertEqual(res.iloc[0]['Position'], 0)
#         self.assertEqual(int(res.iloc[0]['Membrane signal']), int(25631.152970))
#         self.assertEqual(int(res.iloc[0]['Cytoplasmic signal']), int(9235.214863))
#         self.assertEqual(int(iq.roi[0][0, 0]), int(286.61168018921785))
#         # Weirdly, when DE tests are uncommented, this test either fails of doesn't run at all


# class DeCorrectTests(unittest.TestCase):
#     """
#     Making sure results from differential evolution optimiser are as expected
#     Note: if underlying algorithm is changed, or default parameters are changed, tests may fail
#
#     NOTE this will fail due to nondeterministic DE function. Haven't yet found a way to set scipy seed
#
#     """
#
#     # def test1(self):
#     #     # Correct results when quantifying image 1
#     #     iq = ImageQuant(img=imgs[0], roi=rois[0], method='DE', verbose=False)
#     #     iq.run()
#     #     res = iq.compile_res()
#     #     self.assertEqual(res.iloc[0]['Frame'], 0)
#     #     self.assertEqual(res.iloc[0]['Position'], 0)
#     #     self.assertEqual(int(res.iloc[0]['Membrane signal']), int(25636.510450))
#     #     self.assertEqual(int(res.iloc[0]['Cytoplasmic signal']), int(9383.178431))
#     #     self.assertEqual(int(iq.roi[0][0, 0]), int(286.5988590949836))


if __name__ == '__main__':
    unittest.main()
