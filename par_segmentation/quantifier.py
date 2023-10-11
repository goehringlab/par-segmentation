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
        img: Union[np.ndarray, list],
        roi: Union[np.ndarray, list],
        sigma: float = 3.5,
        periodic: bool = True,
        thickness: int = 50,
        rol_ave: int = 5,
        rotate: bool = False,
        nfits: Union[int, None] = 100,
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
        cores: Optional[float] = None,
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
        if self.method == "GD":
            self.sigma = self.iq.sigma

    """
    Saving
    
    """

    def save(self, save_path: str, i: Optional[int] = None):
        """
        Save results for a single image to save_path as a series of txt files and tifs
        I'd recommend using compile_res() instead as this will create a single pandas dataframe with all the results

        Args:
            save_path: path to save full results
            i: index of the image to save (if quantifying multiple images in batch)

        """

        if not self.iq.stack:
            i = 0

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        np.savetxt(
            save_path + "/offsets.txt", self.offsets[i], fmt="%.4f", delimiter="\t"
        )
        np.savetxt(
            save_path + "/cytoplasmic_concentrations.txt",
            self.cyts[i],
            fmt="%.4f",
            delimiter="\t",
        )
        np.savetxt(
            save_path + "/membrane_concentrations.txt",
            self.mems[i],
            fmt="%.4f",
            delimiter="\t",
        )
        np.savetxt(save_path + "/roi.txt", self.roi[i], fmt="%.4f", delimiter="\t")
        save_img(self.target_full[i], save_path + "/target.tif")
        save_img(self.sim_full[i], save_path + "/fit.tif")
        save_img(self.resids_full[i], save_path + "/residuals.tif")

    def compile_res(self):
        """
        Compile results to a pandas dataframe

        Returns:
            A pandas dataframe containing quantification results

        """

        # Assemble dataframe
        _dfs = []
        for i, (m, c) in enumerate(zip(self.mems, self.cyts)):
            _dfs.append(
                pd.DataFrame(
                    {
                        "Frame": i,
                        "Position": range(len(m)),
                        "Membrane signal": m,
                        "Cytoplasmic signal": c,
                    }
                )
            )

        df = pd.concat(_dfs)

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

    def plot_quantification(self):
        """
        Opens an interactive widget to plot membrane quantification results

        """

        jupyter = in_notebook()
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

    def plot_fits(self):
        """
        Opens an interactive widget to plot actual vs fit profiles

        """

        jupyter = in_notebook()
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

    def plot_segmentation(self):
        """
        Opens an interactive widget to plot segmentation results

        """

        jupyter = in_notebook()
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

    def plot_losses(self, log: bool = False):
        """
        Plot loss curves (one line for each image)

        Args:
            log: if True, plot the logarithm of losses

        """

        if self.method == "GD":
            self.iq.plot_losses(log=log)
        else:
            pass
