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
    ):
        """
        Args:
            img: numpy array of image or list of numpy arrays
            roi: coordinates defining the cortex (two column numpy array of x and y
                coordinates at 1-pixel width intervals), or a list of arrays
        """

        # Input data
        self.img = img
        self.roi = roi

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

        # Empty results containers (to be filled by child class)
        self.mems = None
        self.cyts = None
        self.offsets = None
        self.straight_images = None
        self.straight_images_sim = None
        self.straight_images_resids = None

    def save(self, save_path: str, i: int | None = None):
        """
        Save results for a single image to save_path as a series of txt files and tifs
        I'd recommend using compile_res() instead as this will create a single pandas
            dataframe with all the results

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
            "/target.tif": self.straight_images[i],
            "/fit.tif": self.straight_images_sim[i],
            "/residuals.tif": self.straight_images_resids[i],
        }

        for filename, img in image_files.items():
            save_img(img, save_path + filename)

    def compile_res(self, ids=None, extra_columns=None):
        """
        Compile results to a pandas dataframe

        Returns:
            A pandas dataframe containing quantification results
        """
        if ids is None:
            ids = np.arange(self.n)
            index_column_name = "Frame"
        else:
            index_column_name = "EmbryoID"

        # Loop through embryos
        _dfs = []
        for i, (m, c, _id) in enumerate(zip(self.mems, self.cyts, ids)):
            # Construct dictionary
            df_dict = {
                index_column_name: [_id] * len(m),
                "Position": np.arange(len(m)),
                "Membrane signal": m,
                "Cytoplasmic signal": c,
            }

            # Add extra columns
            if extra_columns is not None:
                for key, value in extra_columns.items():
                    df_dict[key] = [value[i] for _ in range(len(m))]

            # Append to list
            _dfs.append(pd.DataFrame(df_dict))

        # Combine
        df = pd.concat(_dfs)

        # Reorder columns
        columns_order = [
            index_column_name,
            "Position",
            "Membrane signal",
            "Cytoplasmic signal",
        ]
        if extra_columns is not None:
            columns_order += list(extra_columns.keys())
        df = df.reindex(columns=columns_order)

        # Specify column types
        df = df.astype({index_column_name: int, "Position": int})
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
        mems_full = self.mems if self.stack else self.mems[0]
        plot_func = plot_quantification_jupyter if jupyter else plot_quantification
        fig, ax = plot_func(mems_full)
        return fig, ax

    def plot_fits(self):
        """
        Opens an interactive widget to plot actual vs fit profiles
        """
        jupyter = in_notebook()
        target_full = self.straight_images if self.stack else self.straight_images[0]
        sim_full = (
            self.straight_images_sim if self.stack else self.straight_images_sim[0]
        )
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
