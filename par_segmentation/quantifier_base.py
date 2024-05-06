from .funcs import in_notebook
from .interactive import (
    view_stack,
    view_stack_jupyter,
    plot_quantification,
    plot_quantification_jupyter,
    plot_fits,
    plot_fits_jupyter,
    plot_segmentation,
    plot_segmentation_jupyter,
)
import os
import numpy as np
import pandas as pd
from .funcs import save_img


class ImageQuantBase:
    def __init__(
        self,
        img: np.ndarray | list,
        roi: np.ndarray | list,
        periodic: bool,
        thickness: int,
        rol_ave: int,
        rotate: bool,
        nfits: int | None,
        zerocap: bool,
        save_training: bool = False,
        save_sims: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            img: numpy array of image or list of numpy arrays
            roi: coordinates defining the cortex (two column numpy array of x and y coordinates at 1-pixel width intervals), or a list of arrays
            periodic: True if coordinates form a closed loop
            thickness: thickness of cross section over which to perform quantification (pixel units)
            rol_ave: width of rolling average to apply to images prior to fitting (pixel units)
            rotate: if True, will automatically rotate ROI so that the first/last points are at the end of the long axis
            nfits: performs this many fits at regular intervals around ROI. If none, will fit at pixel-width intervals
            zerocap: if True, limits output concentrations to positive (or very weakly negative) values
            verbose: False suppresses onscreen output while model is running (e.g. progress bar)
        """

        # Input data
        self.img = img
        self.roi = roi

        # Parameters
        self.periodic = periodic
        self.thickness = thickness
        self.rol_ave = rol_ave
        self.rotate = rotate
        self.nfits = nfits
        self.zerocap = zerocap

        self.save_training = save_training
        self.save_sims = save_sims
        self.verbose = verbose

        # Detect if single frame or stack
        if isinstance(self.img, list) or len(self.img.shape) == 3:
            self.stack = True
            self.img = list(self.img)
        else:
            self.stack = False
            self.img = [self.img]

        self.n = len(self.img)

        # ROI
        if not self.stack:
            self.roi = [self.roi]
        elif isinstance(self.roi, list):
            if len(self.roi) > 1:
                self.roi = self.roi
            else:
                self.roi = self.roi * self.n
        else:
            self.roi = [self.roi] * self.n

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

    def save(self, save_path: str, i: int | None = None):
        """
        Save results for a single image to save_path as a series of txt files and tifs
        I'd recommend using compile_res() instead as this will create a single pandas dataframe with all the results

        Args:
            save_path: path to save full results
            i: index of the image to save (if quantifying multiple images in batch)
        """
        i = 0 if not self.stack else i

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

    def view_frames(self):
        """
        Opens an interactive widget to view image(s)
        """
        jupyter = in_notebook()
        img = self.img if self.stack else self.img[0]
        view_func = view_stack_jupyter if jupyter else view_stack
        fig, ax = view_func(img)
        return fig, ax

    def plot_quantification(self):
        """
        Opens an interactive widget to plot membrane quantification results
        """
        jupyter = in_notebook()
        mems_full = self.mems_full if self.stack else self.mems_full[0]
        plot_func = plot_quantification_jupyter if jupyter else plot_quantification
        fig, ax = plot_func(mems_full)
        return fig, ax

    def plot_fits(self):
        """
        Opens an interactive widget to plot actual vs fit profiles
        """
        jupyter = in_notebook()
        target_full = self.target_full if self.stack else self.target_full[0]
        sim_full = self.sim_full if self.stack else self.sim_full[0]
        plot_func = plot_fits_jupyter if jupyter else plot_fits
        fig, ax = plot_func(target_full, sim_full)
        return fig, ax

    def plot_segmentation(self):
        """
        Opens an interactive widget to plot segmentation results
        """
        jupyter = in_notebook()
        img = self.img if self.stack else self.img[0]
        roi = self.roi if self.stack else self.roi[0]
        plot_func = plot_segmentation_jupyter if jupyter else plot_segmentation
        fig, ax = plot_func(img, roi)
        return fig, ax
