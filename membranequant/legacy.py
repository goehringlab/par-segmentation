import time
import numpy as np
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
import multiprocessing
from .funcs import straighten, rolling_ave_2d, interp_1d_array, interp_2d_array, rotate_roi
from scipy.special import erf
from .roi import interp_roi, offset_coordinates, spline_roi
from typing import Union, Optional, Tuple
import cv2

"""
Legacy code including differential evolution algorithm and other functions no longer used

"""


class ImageQuantDifferentialEvolutionSingle:
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

    def __init__(self,
                 img: Union[np.ndarray, list],
                 sigma: float = 2.0,
                 roi: Union[np.ndarray, list] = None,
                 freedom: float = 0.5,
                 periodic: bool = True,
                 thickness: int = 50,
                 itp: int = 10,
                 rol_ave: int = 10,
                 parallel: bool = False,
                 cores: Optional[int] = None,
                 rotate: bool = False,
                 zerocap: bool = True,
                 nfits: Optional[int] = None,
                 iterations: int = 2,
                 interp: str = 'cubic',
                 bg_subtract: bool = False):

        # Image / stack
        self.img = img

        # ROI
        self.roi_init = roi
        self.roi = roi
        self.periodic = periodic

        # Background subtraction
        self.bg_subtract = bg_subtract

        # Fitting parameters
        self.iterations = iterations
        self.thickness = thickness
        self.itp = itp
        self.thickness_itp = int(itp * self.thickness)
        self.freedom = freedom / (0.5 * thickness)
        self.rol_ave = rol_ave
        self.rotate = rotate
        self.zerocap = zerocap
        self.sigma = sigma
        self.nfits = nfits
        self.interp = interp

        # Background curves
        self.cytbg = (1 + error_func(np.arange(thickness * 2), thickness, (self.sigma * np.sqrt(2)))) / 2
        self.cytbg_itp = (1 + error_func(np.arange(2 * self.thickness_itp), self.thickness_itp,
                                         (self.sigma * np.sqrt(2)) * self.itp)) / 2
        self.membg = gaus(np.arange(thickness * 2), thickness, self.sigma)
        self.membg_itp = gaus(np.arange(2 * self.thickness_itp), self.thickness_itp, self.sigma * self.itp)

        # Computation
        self.parallel = parallel
        if cores is not None:
            self.cores = cores
        else:
            self.cores = multiprocessing.cpu_count()

        # Results containers
        self.offsets = None
        self.cyts = None
        self.mems = None
        self.offsets_full = None
        self.cyts_full = None
        self.mems_full = None

        # Simulated images
        self.straight = None
        self.straight_filtered = None
        self.straight_fit = None
        self.straight_resids = None

        if self.roi is not None:
            self.reset_res()

    """
    Run
    """

    def run(self):

        # Fitting
        for i in range(self.iterations):
            if i > 0:
                self.adjust_roi()
                self.reset_res()
            self.fit()

        # Simulate images
        self.sim_images()

    def fit(self):

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
            self.straight_filtered = rolling_ave_2d(self.straight, self.rol_ave, self.periodic)
        else:
            self.straight_filtered = self.straight

        # Interpolate
        straight = interp_2d_array(self.straight_filtered, self.thickness_itp, method=self.interp)
        straight = interp_2d_array(straight, self.nfits, ax=1, method=self.interp)

        # Fit
        if self.parallel:
            results = np.array(Parallel(n_jobs=self.cores)(
                delayed(self._fit_profile)(straight[:, x]) for x in range(len(straight[0, :]))))
            self.offsets = results[:, 0]
            self.cyts = results[:, 1]
            self.mems = results[:, 2]
        else:
            for x in range(len(straight[0, :])):
                self.offsets[x], self.cyts[x], self.mems[x] = self._fit_profile(straight[:, x])

        # Interpolate
        self.offsets_full = interp_1d_array(self.offsets, len(self.roi[:, 0]), method='linear')
        self.cyts_full = interp_1d_array(self.cyts, len(self.roi[:, 0]), method='linear')
        self.mems_full = interp_1d_array(self.mems, len(self.roi[:, 0]), method='linear')

    def _fit_profile(self, profile: np.ndarray) -> Tuple[float, float, float]:
        if self.zerocap:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (0, max(2 * max(profile), 0)), (0, max(2 * max(profile), 0)))
        else:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (-0.2 * max(profile), 2 * max(profile)), (-0.2 * max(profile), 2 * max(profile)))
        res = differential_evolution(self._mse, bounds=bounds, args=(profile,), tol=0.2)
        o = (res.x[0] - self.thickness_itp / 2) / self.itp
        return o, res.x[1], res.x[2]

    def _mse(self, l_c_m: list, profile: np.ndarray) -> np.ndarray:
        l, c, m = l_c_m
        y = (c * self.cytbg_itp[int(l):int(l) + self.thickness_itp]) + (
                m * self.membg_itp[int(l):int(l) + self.thickness_itp])
        return np.mean((profile - y) ** 2)

    """
    Misc

    """

    def sim_images(self):
        """
        Creates simulated images based on fit results

        """
        for x in range(len(self.roi[:, 0])):
            c = self.cyts_full[x]
            m = self.mems_full[x]
            l = int(self.offsets_full[x] * self.itp + (self.thickness_itp / 2))
            self.straight_fit[:, x] = interp_1d_array(
                (c * self.cytbg_itp[l:l + self.thickness_itp]) + (m * self.membg_itp[l:l + self.thickness_itp]),
                self.thickness, method=self.interp)
            self.straight_resids[:, x] = self.straight[:, x] - self.straight_fit[:, x]

    def adjust_roi(self):
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

    def reset(self):
        """
        Resets entire class to its initial state

        """

        self.roi = self.roi_init
        self.reset_res()

    def reset_res(self):
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


class ImageQuantDifferentialEvolutionMulti:
    def __init__(self,
                 img: Union[np.ndarray, list],
                 roi: Union[np.ndarray, list] = None,
                 sigma: float = 2.0,
                 periodic: bool = True,
                 thickness: int = 50,
                 freedom: float = 0.5,
                 itp: int = 10,
                 rol_ave: int = 10,
                 parallel: bool = False,
                 cores: Optional[int] = None,
                 rotate: bool = False,
                 zerocap: bool = True,
                 nfits: Optional[int] = None,
                 iterations: int = 1,
                 interp: str = 'cubic',
                 bg_subtract: bool = False,
                 verbose: bool = True):

        # Detect if single frame or stack
        if type(img) is list:
            self.stack = True
            self.img = img
        elif len(img.shape) == 3:
            self.stack = True
            self.img = list(img)
        else:
            self.stack = False
            self.img = [img, ]
        self.n = len(self.img)

        # ROI
        if not self.stack:
            self.roi = [roi, ]
        elif type(roi) is list:
            if len(roi) > 1:
                self.roi = roi
            else:
                self.roi = roi * self.n
        else:
            self.roi = [roi] * self.n

        # Set up list of classes
        self.iq = [
            ImageQuantDifferentialEvolutionSingle(img=i, roi=r, sigma=sigma, periodic=periodic, thickness=thickness,
                                                  freedom=freedom, itp=itp, rol_ave=rol_ave, parallel=parallel,
                                                  cores=cores, rotate=rotate, zerocap=zerocap, nfits=nfits,
                                                  iterations=iterations, interp=interp,
                                                  bg_subtract=bg_subtract) for i, r in
            zip(self.img, self.roi)]

        # Initial results containers
        self.mems = [None] * self.n
        self.cyts = [None] * self.n
        self.offsets = [None] * self.n
        self.mems_full = [None] * self.n
        self.cyts_full = [None] * self.n
        self.offsets_full = [None] * self.n
        self.target_full = [None] * self.n
        self.sim_full = [None] * self.n
        self.resids_full = [None] * self.n

        # Verbosity
        self.verbose = verbose

    def run(self):
        t = time.time()

        # Run
        for i, iq in enumerate(self.iq):
            if self.verbose:
                print(f'Quantifying image {i + 1} of {self.n}')
            iq.run()

        # Save membrane/cytoplasmic quantification, offsets
        self.mems[:] = [iq.mems for iq in self.iq]
        self.cyts[:] = [iq.cyts for iq in self.iq]
        self.offsets[:] = [iq.offsets for iq in self.iq]
        self.mems_full[:] = [iq.mems_full for iq in self.iq]
        self.cyts_full[:] = [iq.cyts_full for iq in self.iq]
        self.offsets_full[:] = [iq.offsets_full for iq in self.iq]

        # Save new ROIs
        self.roi[:] = [iq.roi for iq in self.iq]

        # Save target/simulated/residuals images
        self.target_full[:] = [iq.straight_filtered for iq in self.iq]
        self.sim_full[:] = [iq.straight_fit for iq in self.iq]
        self.resids_full[:] = [iq.straight_resids for iq in self.iq]

        if self.verbose:
            print('Time elapsed: %.2f seconds ' % (time.time() - t))


def gaus(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    """
    Create Gaussian curve with centre and width specified

    Args:
        x: array of input x values
        centre: centre of the Gaussian curve (in x units)
        width: width of the gaussian curve (in x units)

    Returns:
        Gaussian curve

    """

    return np.exp(-((x - centre) ** 2) / (2 * width ** 2))


def error_func(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    """
    Create error function with centre and width specified

    Args:
        x: array of input x values
        centre: centre of the error function curve (in x units)
        width: width of the error function curve (in x units)

    Returns:
        error function curve

    """

    return erf((x - centre) / width)


def polycrop(img: np.ndarray, polyline: np.ndarray, enlarge: float) -> np.ndarray:
    """
    Crops image according to polyline coordinates by setting values not contained within the coordinates to zero

    Args:
        img: numpy array of image
        polyline: roi specifying the bounding region (two columns specifying x and y coordinates)
        enlarge: amount by which to expand or contract the ROI (pixel units)

    Returns:
        numpy array of same shape img, with regions outside of polyline set to zero

    """

    newcoors = np.int32(offset_coordinates(polyline, enlarge * np.ones([len(polyline[:, 0])])))
    mask = np.zeros(img.shape)
    mask = cv2.fillPoly(mask, [newcoors], 1)
    newimg = img * mask
    return newimg


def bg_subtraction(img: np.ndarray, roi: np.ndarray, band: tuple = (25, 75)) -> np.ndarray:
    """

    Subtracts background intensity from an image of a cell. Background intensity calculated as the mean intensity within
    a band surronding the cell (specified by ROI)

    Args:
        img: numpy array of image to subtract background from
        roi: two column numpy array specifying coordinates of the cell boundary
        band: inner and outer distance of the band from the roi

    Returns:
        numpy array of image with background subtracted

    """
    a = polycrop(img, roi, band[1]) - polycrop(img, roi, band[0])
    a = [np.nanmean(a[np.nonzero(a)])]
    return img - a


def calc_vol(normcoors: np.ndarray) -> float:
    r1 = (max(normcoors[:, 0]) - min(normcoors[:, 0])) / 2
    r2 = (max(normcoors[:, 1]) - min(normcoors[:, 1])) / 2
    return (4 / 3) * np.pi * r2 * r2 * r1


def calc_sa(normcoors: np.ndarray) -> float:
    r1 = (max(normcoors[:, 0]) - min(normcoors[:, 0])) / 2
    r2 = (max(normcoors[:, 1]) - min(normcoors[:, 1])) / 2
    e = (1 - (r2 ** 2) / (r1 ** 2)) ** 0.5
    return 2 * np.pi * r2 * r2 * (1 + (r1 / (r2 * e)) * np.arcsin(e))
