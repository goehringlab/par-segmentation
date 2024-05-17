import multiprocessing
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution

from .funcs import (
    interp_1d_array,
    interp_2d_array,
    rolling_ave_2d,
    rotate_roi,
    straighten,
    error_func,
    gaus,
)
from .roi import interp_roi, offset_coordinates, spline_roi
from .model_base import ImageQuantBase

"""
Legacy code for differential evolution algorithm 

NO LONGER MAINTAINED

"""


class ImageQuantDifferentialEvolutionSingle(ImageQuantBase):
    """
    Quantification works by taking cross-sections across the membrane, and fitting the resulting profile as the sum of
    a cytoplasmic signal component and a membrane signal component. Differential evolution algorithm

    Input data:
    img                image

    Background curves:
    sigma              if either of above are not specified, assume gaussian/error function with width set by sigma

    ROI:
    roi                coordinates defining cortex. Can use output from def_roi function

    Fitting parameters:
    freedom            amount of freedom allowed in ROI (pixel units)
    periodic           True if coordinates form a closed loop
    thickness          thickness of cross-section over which to perform quantification
    itp                amount to interpolate image prior to segmentation (this many points per pixel in original image)
    rol_ave            width of rolling average
    rotate             if True, will automatically rotate ROI so that the first/last points are at the end of the long
                       axis
    zerocap            if True, prevents negative membane and cytoplasm values
    nfits              performs this many fits at regular intervals around ROI
    iterations         if >1, adjusts ROI and re-fits
    interp             interpolation type (linear or cubic)
    bg_subtract        if True, will estimate and subtract background signal prior to quantification

    Computation:
    parallel           TRUE = perform fitting in parallel
    cores              number of cores to use if parallel is True (if none will use all available)

    Saving:
    save_path          destination to save results, will create if it doesn't already exist


    """

    def __init__(
        self,
        img: np.ndarray | list,
        roi: np.ndarray | list = None,
        sigma: float = 3.5,
        freedom: float = 25,
        periodic: bool = True,
        thickness: int = 50,
        itp: int = 10,
        rol_ave: int = 5,
        parallel: bool = False,
        cores: int | None = None,
        rotate: bool = False,
        zerocap: bool = False,
        nfits: int | None = 100,
        iterations: int = 2,
        interp: str = "cubic",
        bg_subtract: bool = False,
    ):
        super().__init__(
            img=img,
            roi=roi,
        )
        self.img = self.img[0]
        self.roi = self.roi[0]
        self.roi_init = self.roi

        # Model parameters
        self.periodic = periodic
        self.thickness = thickness
        self.rol_ave = rol_ave
        self.rotate = rotate
        self.nfits = nfits
        self.zerocap = zerocap
        self.bg_subtract = bg_subtract
        self.iterations = iterations
        self.itp = itp
        self.thickness_itp = int(itp * self.thickness)
        self.freedom = freedom / (0.5 * thickness)
        self.sigma = sigma
        self.interp = interp

        # Background curves
        self.cytbg = (
            1
            + error_func(np.arange(thickness * 2), thickness, (self.sigma * np.sqrt(2)))
        ) / 2
        self.cytbg_itp = (
            1
            + error_func(
                np.arange(2 * self.thickness_itp),
                self.thickness_itp,
                (self.sigma * np.sqrt(2)) * self.itp,
            )
        ) / 2
        self.membg = gaus(np.arange(thickness * 2), thickness, self.sigma)
        self.membg_itp = gaus(
            np.arange(2 * self.thickness_itp), self.thickness_itp, self.sigma * self.itp
        )

        # Computation
        self.parallel = parallel
        if cores is not None:
            self.cores = cores
        else:
            self.cores = multiprocessing.cpu_count()

        # Simulated images
        self.straight = None
        self.straight_filtered = None
        self.straight_fit = None
        self.straight_resids = None

        # Interpolated results
        self.mems_full = None
        self.cyts_full = None
        self.offsets_full = None

        if self.roi is not None:
            self._reset_res()

    """
    Run
    """

    def run(self):
        # Fitting
        for i in range(self.iterations):
            if i > 0:
                self._adjust_roi()
                self._reset_res()
            self._fit()

        # Simulate images
        self._sim_images()

    def _fit(self):
        # Specify number of fits
        if self.nfits is None:
            self.nfits = len(self.roi[:, 0])

        # Straighten image
        self.straight = straighten(self.img, self.roi, self.thickness)

        # Background subtract
        if self.bg_subtract:
            bg_intensity = np.mean(self.straight[:5, :])
            self.straight -= bg_intensity
            self.img -= bg_intensity

        # Smoothen
        if self.rol_ave != 0:
            self.straight_filtered = rolling_ave_2d(
                self.straight, self.rol_ave, self.periodic
            )
        else:
            self.straight_filtered = self.straight

        # Interpolate
        straight = interp_2d_array(
            self.straight_filtered, self.thickness_itp, method=self.interp
        )
        straight = interp_2d_array(straight, self.nfits, ax=1, method=self.interp)

        # Fit
        if self.parallel:
            results = np.array(
                Parallel(n_jobs=self.cores)(
                    delayed(self._fit_profile)(straight[:, x])
                    for x in range(len(straight[0, :]))
                )
            )
            self.offsets = results[:, 0]
            self.cyts = results[:, 1]
            self.mems = results[:, 2]
        else:
            for x in range(len(straight[0, :])):
                self.offsets[x], self.cyts[x], self.mems[x] = self._fit_profile(
                    straight[:, x]
                )

        # Interpolate
        self.offsets_full = interp_1d_array(
            self.offsets, len(self.roi[:, 0]), method="linear"
        )
        self.cyts_full = interp_1d_array(
            self.cyts, len(self.roi[:, 0]), method="linear"
        )
        self.mems_full = interp_1d_array(
            self.mems, len(self.roi[:, 0]), method="linear"
        )

    def _fit_profile(self, profile: np.ndarray) -> tuple[float, float, float]:
        if self.zerocap:
            bounds = (
                (
                    (self.thickness_itp / 2) * (1 - self.freedom),
                    (self.thickness_itp / 2) * (1 + self.freedom),
                ),
                (0, max(2 * max(profile), 0)),
                (0, max(2 * max(profile), 0)),
            )
        else:
            bounds = (
                (
                    (self.thickness_itp / 2) * (1 - self.freedom),
                    (self.thickness_itp / 2) * (1 + self.freedom),
                ),
                (-0.2 * max(profile), 2 * max(profile)),
                (-0.2 * max(profile), 2 * max(profile)),
            )
        res = differential_evolution(self._mse, bounds=bounds, args=(profile,), tol=0.2)
        o = (res.x[0] - self.thickness_itp / 2) / self.itp
        return o, res.x[1], res.x[2]

    def _mse(self, l_c_m: list, profile: np.ndarray) -> np.ndarray:
        slice_index, c, m = l_c_m
        y = (
            c * self.cytbg_itp[int(slice_index) : int(slice_index) + self.thickness_itp]
        ) + (
            m * self.membg_itp[int(slice_index) : int(slice_index) + self.thickness_itp]
        )
        return np.mean((profile - y) ** 2)

    """
    Misc

    """

    def _sim_images(self):
        """
        Creates simulated images based on fit results

        """
        for x in range(len(self.roi[:, 0])):
            c = self.cyts_full[x]
            m = self.mems_full[x]
            slice_index = int(
                self.offsets_full[x] * self.itp + (self.thickness_itp / 2)
            )
            self.straight_fit[:, x] = interp_1d_array(
                (c * self.cytbg_itp[slice_index : slice_index + self.thickness_itp])
                + (m * self.membg_itp[slice_index : slice_index + self.thickness_itp]),
                self.thickness,
                method=self.interp,
            )
            self.straight_resids[:, x] = self.straight[:, x] - self.straight_fit[:, x]

    def _adjust_roi(self):
        """
        Can do after a preliminary fit to refine coordinates
        Must refit after doing this

        """

        # Offset coordinates
        self.roi = offset_coordinates(self.roi, self.offsets_full)

        # Fit spline
        self.roi = spline_roi(roi=self.roi, periodic=self.periodic, s=100)

        # Interpolate to one px distance between points
        self.roi = interp_roi(self.roi, self.periodic)

        # Rotate
        if self.periodic:
            if self.rotate:
                self.roi = rotate_roi(self.roi)

    def _reset(self):
        """
        Resets entire class to its initial state

        """

        self.roi = self.roi_init
        self._reset_res()

    def _reset_res(self):
        """
        Clears results

        """

        if self.nfits is None:
            self.nfits = len(self.roi[:, 0])

        # Results
        self.offsets = np.zeros(self.nfits)
        self.cyts = np.zeros(self.nfits)
        self.mems = np.zeros(self.nfits)

        # Interpolated results
        self.offsets_full = np.zeros(len(self.roi[:, 0]))
        self.cyts_full = np.zeros(len(self.roi[:, 0]))
        self.mems_full = np.zeros(len(self.roi[:, 0]))

        # Simulated images
        self.straight = np.zeros([self.thickness, len(self.roi[:, 0])])
        self.straight_filtered = np.zeros([self.thickness, len(self.roi[:, 0])])
        self.straight_fit = np.zeros([self.thickness, len(self.roi[:, 0])])
        self.straight_resids = np.zeros([self.thickness, len(self.roi[:, 0])])


class ImageQuantDifferentialEvolutionMulti(ImageQuantBase):
    def __init__(
        self,
        img: np.ndarray | list,
        roi: np.ndarray | list = None,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(
            img=img,
            roi=roi,
        )
        self.verbose = verbose

        # Set up list of classes
        self.iq = [
            ImageQuantDifferentialEvolutionSingle(
                img=i,
                roi=r,
                **kwargs,
            )
            for i, r in zip(self.img, self.roi)
        ]

    def run(self):
        t = time.time()

        # Run
        for i, iq in enumerate(self.iq):
            if self.verbose:
                print(f"Quantifying image {i + 1} of {self.n}")
            iq.run()

        # Save membrane/cytoplasmic quantification, offsets
        self.mems = [iq.mems for iq in self.iq]
        self.cyts = [iq.cyts for iq in self.iq]
        self.offsets = [iq.offsets for iq in self.iq]
        self.mems_full = [iq.mems_full for iq in self.iq]
        self.cyts_full = [iq.cyts_full for iq in self.iq]
        self.offsets_full = [iq.offsets_full for iq in self.iq]

        # Save new ROIs
        self.roi = [iq.roi for iq in self.iq]

        # Save target/simulated/residuals images
        self.straight_images = [iq.straight_filtered for iq in self.iq]
        self.straight_images_sim = [iq.straight_fit for iq in self.iq]
        self.straight_images_resids = [iq.straight_resids for iq in self.iq]

        if self.verbose:
            print("Time elapsed: %.2f seconds " % (time.time() - t))
