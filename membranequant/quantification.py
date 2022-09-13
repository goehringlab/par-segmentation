import numpy as np
import os
import pandas as pd
from .funcs import save_img
from .interactive import view_stack, view_stack_jupyter, plot_fits, plot_fits_jupyter, plot_segmentation, \
    plot_segmentation_jupyter, plot_quantification, plot_quantification_jupyter
from .quantificationGradientDescent import ImageQuantGradientDescent
from .quantificationDifferentialEvolutionMulti import ImageQuantDifferentialEvolutionMulti


class ImageQuant:
    def __init__(self, img, roi, sigma=2, periodic=True, thickness=50, rol_ave=10, rotate=False, nfits=100,
                 iterations=2, lr=0.01, descent_steps=500, adaptive_sigma=False, batch_norm=False, freedom=10,
                 roi_knots=20, fit_outer=False, save_training=False, save_sims=False, method='GD', itp=10,
                 parallel=False, zerocap=False, cores=None, bg_subtract=False, interp='cubic'):

        # Input data
        self.img = img
        self.roi = roi

        # Set up quantifier
        self.method = method
        if self.method == 'GD':
            self.iq = ImageQuantGradientDescent(img=img, roi=roi, sigma=sigma, periodic=periodic, thickness=thickness,
                                                rol_ave=rol_ave, rotate=rotate, nfits=nfits, iterations=iterations,
                                                lr=lr, descent_steps=descent_steps, adaptive_sigma=adaptive_sigma,
                                                batch_norm=batch_norm, freedom=freedom, roi_knots=roi_knots,
                                                fit_outer=fit_outer, save_training=save_training, save_sims=save_sims)

        elif self.method == 'DE':
            self.iq = ImageQuantDifferentialEvolutionMulti(img=img, roi=roi, sigma=sigma, periodic=periodic,
                                                           thickness=thickness, freedom=freedom, itp=itp,
                                                           rol_ave=rol_ave, parallel=parallel, cores=cores,
                                                           rotate=rotate, zerocap=zerocap, nfits=nfits,
                                                           iterations=iterations, interp=interp,
                                                           bg_subtract=bg_subtract)
        else:
            raise Exception('Method must be "GD" (gradient descent) or "DE" (differential evolution)')

        # Empty results containers
        self.mems = None
        self.cyts = None
        self.offsets = None
        self.mems_full = None
        self.cyts_full = None
        self.offsets_full = None
        self.target_full = None
        self.sim_full = None
        self.resids_full = None

    def run(self):
        self.iq.run()

        # Save new ROI
        self.roi = self.iq.roi

        # Save results
        self.mems = self.iq.mems
        self.cyts = self.iq.cyts
        self.offsets = self.iq.offsets
        self.mems_full = self.iq.mems_full
        self.cyts_full = self.iq.cyts_full
        self.offsets_full = self.iq.offsets_full
        self.target_full = self.iq.target_full
        self.sim_full = self.iq.sim_full
        self.resids_full = self.iq.resids_full

    def save(self, save_path, i=None):
        """
        Save all results to save_path

        """

        if not self.iq.stack:
            i = 0

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        np.savetxt(save_path + '/offsets.txt', self.offsets[i], fmt='%.4f', delimiter='\t')
        np.savetxt(save_path + '/cytoplasmic_concentrations.txt', self.cyts[i], fmt='%.4f', delimiter='\t')
        np.savetxt(save_path + '/membrane_concentrations.txt', self.mems[i], fmt='%.4f', delimiter='\t')
        np.savetxt(save_path + '/roi.txt', self.roi[i], fmt='%.4f', delimiter='\t')
        save_img(self.target_full[i], save_path + '/target.tif')
        save_img(self.sim_full[i], save_path + '/fit.tif')
        save_img(self.resids_full[i], save_path + '/residuals.tif')

    def compile_res(self):
        # Create empty dataframe
        df = pd.DataFrame({'Frame': [],
                           'Position': [],
                           'Membrane signal': [],
                           'Cytoplasmic signal': []})

        # Fill with data
        for i, (m, c) in enumerate(zip(self.mems, self.cyts)):
            df = df.append(pd.DataFrame({'Frame': i,
                                         'Position': range(len(m)),
                                         'Membrane signal': m,
                                         'Cytoplasmic signal': c}))

        df = df.reindex(columns=['Frame', 'Position', 'Membrane signal', 'Cytoplasmic signal'])
        df = df.astype({'Frame': int, 'Position': int})
        return df

    def view_frames(self, jupyter=False):
        if not jupyter:
            if self.iq.stack:
                fig, ax = view_stack(self.img)
            else:
                fig, ax = view_stack(self.img[0])
        else:
            if self.iq.stack:
                fig, ax = view_stack_jupyter(self.img)
            else:
                fig, ax = view_stack_jupyter(self.img[0])
        return fig, ax

    def plot_quantification(self, jupyter=False):
        if not jupyter:
            if self.iq.stack:
                fig, ax = plot_quantification(self.mems_full)
            else:
                fig, ax = plot_quantification(self.mems_full[0])
        else:
            if self.iq.stack:
                fig, ax = plot_quantification_jupyter(self.mems_full)
            else:
                fig, ax = plot_quantification_jupyter(self.mems_full[0])
        return fig, ax

    def plot_fits(self, jupyter=False):
        if not jupyter:
            if self.iq.stack:
                fig, ax = plot_fits(self.target_full, self.sim_full)
            else:
                fig, ax = plot_fits(self.target_full[0], self.sim_full[0])
        else:
            if self.iq.stack:
                fig, ax = plot_fits_jupyter(self.target_full, self.sim_full)
            else:
                fig, ax = plot_fits_jupyter(self.target_full[0], self.sim_full[0])
        return fig, ax

    def plot_segmentation(self, jupyter=False):
        if not jupyter:
            if self.iq.stack:
                fig, ax = plot_segmentation(self.img, self.roi)
            else:
                fig, ax = plot_segmentation(self.img[0], self.roi[0])
        else:
            if self.iq.stack:
                fig, ax = plot_segmentation_jupyter(self.img, self.roi)
            else:
                fig, ax = plot_segmentation_jupyter(self.img[0], self.roi[0])
        return fig, ax

    def plot_losses(self, log=False):
        if self.method == 'GD':
            self.iq.plot_losses(log=log)
        else:
            pass
