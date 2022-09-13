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

    def run(self):
        self.iq.run()

    def save(self, save_path, i=None):
        """
        Save all results to save_path

        """

        if not self.iq.stack:
            i = 0

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        np.savetxt(save_path + '/offsets.txt', self.iq.offsets[i], fmt='%.4f', delimiter='\t')
        np.savetxt(save_path + '/cytoplasmic_concentrations.txt', self.iq.cyts[i], fmt='%.4f', delimiter='\t')
        np.savetxt(save_path + '/membrane_concentrations.txt', self.iq.mems[i], fmt='%.4f', delimiter='\t')
        np.savetxt(save_path + '/roi.txt', self.iq.roi[i], fmt='%.4f', delimiter='\t')
        save_img(self.iq.target_full[i], save_path + '/target.tif')
        save_img(self.iq.sim_full[i], save_path + '/fit.tif')
        save_img(self.iq.resids_full[i], save_path + '/residuals.tif')

    def compile_res(self):
        # Create empty dataframe
        df = pd.DataFrame({'Frame': [],
                           'Position': [],
                           'Membrane signal': [],
                           'Cytoplasmic signal': []})

        # Fill with data
        for i, (m, c) in enumerate(zip(self.iq.mems, self.iq.cyts)):
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
                fig, ax = view_stack(self.iq.img)
            else:
                fig, ax = view_stack(self.iq.img[0])
        else:
            if self.iq.stack:
                fig, ax = view_stack_jupyter(self.iq.img)
            else:
                fig, ax = view_stack_jupyter(self.iq.img[0])
        return fig, ax

    def plot_quantification(self, jupyter=False):
        if not jupyter:
            if self.iq.stack:
                fig, ax = plot_quantification(self.iq.mems_full)
            else:
                fig, ax = plot_quantification(self.iq.mems_full[0])
        else:
            if self.iq.stack:
                fig, ax = plot_quantification_jupyter(self.iq.mems_full)
            else:
                fig, ax = plot_quantification_jupyter(self.iq.mems_full[0])
        return fig, ax

    def plot_fits(self, jupyter=False):
        if not jupyter:
            if self.iq.stack:
                fig, ax = plot_fits(self.iq.target_full, self.iq.sim_full)
            else:
                fig, ax = plot_fits(self.iq.target_full[0], self.iq.sim_full[0])
        else:
            if self.iq.stack:
                fig, ax = plot_fits_jupyter(self.iq.target_full, self.iq.sim_full)
            else:
                fig, ax = plot_fits_jupyter(self.iq.target_full[0], self.iq.sim_full[0])
        return fig, ax

    def plot_segmentation(self, jupyter=False):
        if not jupyter:
            if self.iq.stack:
                fig, ax = plot_segmentation(self.iq.img, self.iq.roi)
            else:
                fig, ax = plot_segmentation(self.iq.img[0], self.iq.roi[0])
        else:
            if self.iq.stack:
                fig, ax = plot_segmentation_jupyter(self.iq.img, self.iq.roi)
            else:
                fig, ax = plot_segmentation_jupyter(self.iq.img[0], self.iq.roi[0])
        return fig, ax

    def plot_losses(self, log=False):
        if self.method == 'GD':
            self.iq.plot_losses(log=log)
        else:
            pass
