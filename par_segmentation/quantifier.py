import os
from typing import Optional, Union

import numpy as np
import pandas as pd

from .funcs import in_notebook, save_img
from .interactive import (
    plot_fits,
    plot_fits_jupyter,
    plot_quantification,
    plot_quantification_jupyter,
    plot_segmentation,
    plot_segmentation_jupyter,
    view_stack,
    view_stack_jupyter,
)
from .legacy import ImageQuantDifferentialEvolutionMulti
from .model import ImageQuantGradientDescent

__all__ = ["ImageQuant"]


class ImageQuant:
    """

    Main class to perform image segmentation

    Instructions:\n
    1. (Optional) Perform SAIBR on image\n
    2. Specify rough manual ROI\n
    3. Initialise class\n
    4. run()\n
    5. New ROI coordinates will be found at self.roi\n
    6. Save quantification results using compile_res() - returns a pandas dataframe

    Args:
        img: numpy array of image or list of numpy arrays
        roi: coordinates defining the cortex (two column numpy array of x and y coordinates at 1-pixel width intervals), or a list of arrays
        roi_knots: number of knots in cubic-spline fit ROI
        freedom: amount by which the roi can move (pixel units)
        sigma: gaussian/error function width (pixels units)
        periodic: True if coordinates form a closed loop
        thickness: thickness of cross section over which to perform quantification (pixel units)
        rol_ave: width of rolling average to apply to images prior to fitting (pixel units)
        rotate: if True, will automatically rotate ROI so that the first/last points are at the end of the long axis
        nfits: performs this many fits at regular intervals around ROI. If none, will fit at pixel-width intervals
        iterations: if >1, adjusts ROI and re-fits
        batch_norm: if True, images will be globally, rather than internally, normalised. Shouldn't affect quantification but is recommended during model optimisation
        fit_outer: if True, will fit the outer portion of each profile to a nonzero value
        method: 'GD' for gradient descent or 'DE' for differential evolution. The former is highly recommended, the latter works but is much slower and no longer maintained
        zerocap: if True, limits output concentrations to positive (or very weakly negative) values
        interp: interpolation type, 'cubic' or 'linear'
        lr: learning rate
        descent_steps: number of gradient descent steps
        adaptive_sigma: if True, sigma will be trained by gradient descent
        verbose: False suppresses onscreen output while model is running (e.g. progress bar)
        parallel: LEGACY (for DE method). If True will run in parallel on number of cores specified. NB Very buggy
        cores:  LEGACY (for DE method). Number of cores to use if parallel is True
        itp: LEGACY (for DE method). Amount of interpolation - allows for subpixel alignment

    """

    def __init__(
        self,
        img: np.ndarray | list,
        roi: np.ndarray | list,
        sigma: float = 3.5,
        periodic: bool = True,
        thickness: int = 50,
        rol_ave: int = 5,
        rotate: bool = False,
        nfits: int | None = 100,
        iterations: int = 2,
        lr: float = 0.01,
        descent_steps: int = 400,
        adaptive_sigma: bool = False,
        batch_norm: bool = False,
        freedom: float = 25,
        roi_knots: int = 20,
        fit_outer: bool = True,
        save_training: bool = False,
        save_sims: bool = False,
        method: str = "GD",
        itp: int = 10,
        parallel: bool = False,
        zerocap: bool = False,
        cores: float | None = None,
        bg_subtract: bool = False,
        interp: str = "cubic",
        verbose: bool = True,
    ):
        # Set up quantifier
        self.method = method
        if self.method == "GD":
            self.iq = ImageQuantGradientDescent(
                img=img,
                roi=roi,
                sigma=sigma,
                periodic=periodic,
                thickness=thickness,
                rol_ave=rol_ave,
                rotate=rotate,
                nfits=nfits,
                iterations=iterations,
                lr=lr,
                descent_steps=descent_steps,
                adaptive_sigma=adaptive_sigma,
                batch_norm=batch_norm,
                freedom=freedom,
                roi_knots=roi_knots,
                fit_outer=fit_outer,
                zerocap=zerocap,
                save_training=save_training,
                save_sims=save_sims,
                verbose=verbose,
            )

        elif self.method == "DE":
            self.iq = ImageQuantDifferentialEvolutionMulti(
                img=img,
                roi=roi,
                sigma=sigma,
                periodic=periodic,
                thickness=thickness,
                freedom=freedom,
                itp=itp,
                rol_ave=rol_ave,
                parallel=parallel,
                cores=cores,
                rotate=rotate,
                zerocap=zerocap,
                nfits=nfits,
                iterations=iterations,
                interp=interp,
                bg_subtract=bg_subtract,
                verbose=verbose,
            )
        else:
            raise Exception(
                'Method must be "GD" (gradient descent) or "DE" (differential evolution)'
            )

        # Input data
        self.img = self.iq.img
        self.roi = self.iq.roi

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
        self.sigma = None

    """
    Run
    
    """

    def run(self):
        """
        Performs segmentation/quantification and saves results
        """
        self.iq.run()

        # Save results
        attributes = [
            "roi",
            "mems",
            "cyts",
            "offsets",
            "mems_full",
            "cyts_full",
            "offsets_full",
            "target_full",
            "sim_full",
            "resids_full",
        ]

        if self.method == "GD":
            attributes.append("sigma")

        for attr in attributes:
            setattr(self, attr, getattr(self.iq, attr))

    """
    Saving
    
    """

    def save(self, save_path: str, i: int | None = None):
        """
        Save results for a single image to save_path as a series of txt files and tifs
        I'd recommend using compile_res() instead as this will create a single pandas dataframe with all the results

        Args:
            save_path: path to save full results
            i: index of the image to save (if quantifying multiple images in batch)
        """
        i = 0 if not self.iq.stack else i

        os.makedirs(save_path, exist_ok=True)

        data_files = {
            "/offsets.txt": self.offsets[i],
            "/cytoplasmic_concentrations.txt": self.cyts[i],
            "/membrane_concentrations.txt": self.mems[i],
            "/roi.txt": self.roi[i],
        }

        for filename, data in data_files.items():
            np.savetxt(save_path + filename, data, fmt="%.4f", delimiter="\t")

        image_files = {
            "/target.tif": self.target_full[i],
            "/fit.tif": self.sim_full[i],
            "/residuals.tif": self.resids_full[i],
        }

        for filename, img in image_files.items():
            save_img(img, save_path + filename)

    def compile_res(self):
        """
        Compile results to a pandas dataframe

        Returns:
            A pandas dataframe containing quantification results
        """
        # Assemble dataframe
        df = pd.concat(
            pd.DataFrame(
                {
                    "Frame": i,
                    "Position": range(len(m)),
                    "Membrane signal": m,
                    "Cytoplasmic signal": c,
                }
            )
            for i, (m, c) in enumerate(zip(self.mems, self.cyts))
        )

        # Tidy up
        df = df.reindex(
            columns=["Frame", "Position", "Membrane signal", "Cytoplasmic signal"]
        )
        df = df.astype({"Frame": int, "Position": int})
        return df

    """
    Interactive
    
    """

    def view_frames(self):
        """
        Opens an interactive widget to view image(s)
        """
        jupyter = in_notebook()
        img = self.img if self.iq.stack else self.img[0]
        view_func = view_stack_jupyter if jupyter else view_stack
        fig, ax = view_func(img)
        return fig, ax

    def plot_quantification(self):
        """
        Opens an interactive widget to plot membrane quantification results
        """
        jupyter = in_notebook()
        mems_full = self.mems_full if self.iq.stack else self.mems_full[0]
        plot_func = plot_quantification_jupyter if jupyter else plot_quantification
        fig, ax = plot_func(mems_full)
        return fig, ax

    def plot_fits(self):
        """
        Opens an interactive widget to plot actual vs fit profiles
        """
        jupyter = in_notebook()
        target_full = self.target_full if self.iq.stack else self.target_full[0]
        sim_full = self.sim_full if self.iq.stack else self.sim_full[0]
        plot_func = plot_fits_jupyter if jupyter else plot_fits
        fig, ax = plot_func(target_full, sim_full)
        return fig, ax

    def plot_segmentation(self):
        """
        Opens an interactive widget to plot segmentation results
        """
        jupyter = in_notebook()
        img = self.img if self.iq.stack else self.img[0]
        roi = self.roi if self.iq.stack else self.roi[0]
        plot_func = plot_segmentation_jupyter if jupyter else plot_segmentation
        fig, ax = plot_func(img, roi)
        return fig, ax

    def plot_losses(self, log: bool = False):
        """
        Plot loss curves (one line for each image)

        Args:
            log: if True, plot the logarithm of losses
        """
        if self.method == "GD":
            self.iq.plot_losses(log=log)
