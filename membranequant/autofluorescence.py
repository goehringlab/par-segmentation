import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import random
import glob
import scipy.odr as odr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from .funcs import load_image, make_mask
from .roi import offset_coordinates
from typing import Tuple, Optional


class AfCorrelation:
    def __init__(self,
                 paths: list,
                 gfp_regex: str = '*488 SP 535-50*',
                 af_regex: str = '*488 SP 630-75*',
                 rfp_regex: Optional[str] = None,
                 roi_regex: str = '*ROI*',
                 sigma: float = 2.0,
                 intercept0: bool = False,
                 expand: float = 5.0,
                 method: str = 'OLS'):

        """

        Convenient class for performing SAIBR calibration
        Imports data according to paths and regular expressions, blurs images, and performs the regression

        Args:
            paths: list of paths containing n2 images
            gfp_regex: regular expression found in gfp channel image files
            af_regex: regular expression found in af channel image files
            rfp_regex: regular expression found in rfp channel files (optional)
            roi_regex: regular exxpression found in ROI files
            sigma: gaussian blur to apply to images prior to regression
            intercept0: if True, force intercept of regression to go through zero. Not recommended
            expand: expand ROIs by this many pixels (useful to include a portion of background)
            method: fitting method, either 'OLS' for ordinary least squares or 'ODR' for orthogonal distance regression

        To run regression, initialise class and run()
        SAIBR parameters will then be found at self.params

        """

        # Global parameters
        self.sigma = sigma
        self.intercept0 = intercept0
        self.method = method

        # Import images
        self.gfp = [load_image(sorted(glob.glob(f'{p}/{gfp_regex}'))[0]) for p in paths]
        self.af = [load_image(sorted(glob.glob(f'{p}/{af_regex}'))[0]) for p in paths]
        if rfp_regex is not None:
            self.rfp = [load_image(sorted(glob.glob(f'{p}/{rfp_regex}'))[0]) for p in paths]
        else:
            self.rfp = None

        # Import rois
        if roi_regex is not None:
            self.roi = [offset_coordinates(np.loadtxt(sorted(glob.glob(f'{p}/{roi_regex}'))[0]), expand) for p in
                        paths]
        else:
            self.roi = None

        # Create masks
        if self.roi is not None:
            self.mask = [make_mask([512, 512], r) for r in self.roi]
        else:
            self.mask = None

        # Apply filters
        self.gfp_filtered = [gaussian_filter(c, sigma=self.sigma) for c in self.gfp]
        self.af_filtered = [gaussian_filter(c, sigma=self.sigma) for c in self.af]
        if self.rfp is not None:
            self.rfp_filtered = [gaussian_filter(c, sigma=self.sigma) for c in self.rfp]

        # Results
        self.params = None
        self.gfp_vals = None
        self.af_vals = None
        self.rfp_vals = None
        self.r2 = None

    def run(self):

        # Perform regression
        if self.rfp is None:
            self.params, self.af_vals, self.gfp_vals = af_correlation(np.array(self.gfp_filtered),
                                                                      np.array(self.af_filtered), self.mask,
                                                                      intercept0=self.intercept0,
                                                                      method=self.method)
        else:
            self.params, self.af_vals, self.rfp_vals, self.gfp_vals = af_correlation_3channel(
                np.array(self.gfp_filtered), np.array(self.af_filtered), np.array(self.rfp_filtered), self.mask,
                intercept0=self.intercept0, method=self.method)

        # Calculate ypred
        if self.rfp is None:
            ypred = self.params[0] * self.af_vals + self.params[1]
        else:
            ypred = self.params[0] * self.af_vals + self.params[1] * self.rfp_vals + self.params[2]

        # Calculate R squared
        self.r2 = r2_score(self.gfp_vals, ypred)

    def plot_correlation(self, s: Optional[float] = None):
        if self.rfp is None:
            s = 0.001 if s is None else s
            fig, ax = self._plot_correlation_2channel(s=s)
        else:
            s = 0.1 if s is None else s
            fig, ax = self._plot_correlation_3channel(s=s)
        return fig, ax

    def _plot_correlation_2channel(self, s: float = 0.001):
        fig, ax = plt.subplots()

        # Scatter
        for c1, c2, m in zip(self.gfp_filtered, self.af_filtered, self.mask):
            c1_masked = c1 * m
            c2_masked = c2 * m
            c1_vals = c1_masked[~np.isnan(c1_masked)]
            c2_vals = c2_masked[~np.isnan(c2_masked)]
            ax.scatter(c2_vals, c1_vals, s=s)

        # Plot line
        xline = np.linspace(np.percentile(self.af_vals, 0.01), np.percentile(self.af_vals, 99.99), 20)
        yline = self.params[0] * xline + self.params[1]
        ax.plot(xline, yline, c='k', linestyle='--')

        # Finalise figure
        ax.set_xlim(np.percentile(self.af_vals, 0.01), np.percentile(self.af_vals, 99.99))
        ax.set_ylim(np.percentile(self.gfp_vals, 0.01), np.percentile(self.gfp_vals, 99.99))
        ax.set_xlabel('AF')
        ax.set_ylabel('GFP')
        return fig, ax

    def _plot_correlation_3channel(self, s: float = 1.0):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        xx, yy = np.meshgrid([np.percentile(self.af_vals, 0.01), np.percentile(self.af_vals, 99.99)],
                             [np.percentile(self.rfp_vals, 0.01), np.percentile(self.rfp_vals, 99.99)])
        zz = self.params[0] * xx + self.params[1] * yy + self.params[2]
        ax.plot_surface(xx, yy, zz, alpha=0.2)

        # Scatter plot
        for c1, c2, c3, m in zip(self.gfp_filtered, self.af_filtered, self.rfp_filtered, self.mask):
            c1_masked = c1 * m
            c2_masked = c2 * m
            c3_masked = c3 * m
            c1_vals = c1_masked[~np.isnan(c1_masked)]
            c2_vals = c2_masked[~np.isnan(c2_masked)]
            c3_vals = c3_masked[~np.isnan(c3_masked)]
            sample = random.sample(range(len(c1_vals)), min(1000, len(c1_vals)))
            ax.scatter(c2_vals[sample], c3_vals[sample], c1_vals[sample], s=s)

        # Finalise figure
        ax.set_xlim(np.percentile(self.af_vals, 0.01), np.percentile(self.af_vals, 99.99))
        ax.set_ylim(np.percentile(self.rfp_vals, 0.01), np.percentile(self.rfp_vals, 99.99))
        ax.set_zlim(np.percentile(self.gfp_vals, 0.01), np.percentile(self.gfp_vals, 99.99))
        ax.set_xlabel('AF')
        ax.set_ylabel('RFP')
        ax.set_zlabel('GFP')
        return fig, ax

    def plot_prediction(self, s: float = 0.001):
        if self.rfp is None:
            fig, ax = self._plot_prediction_2channel(s=s)
        else:
            fig, ax = self._plot_prediction_3channel(s=s)
        return fig, ax

    def _plot_prediction_2channel(self, s: float = 0.001):
        fig, ax = plt.subplots()

        # Scatter
        for c1, c2, m in zip(self.gfp_filtered, self.af_filtered, self.mask):
            c1_masked = c1 * m
            c2_masked = c2 * m
            c1_vals = c1_masked[~np.isnan(c1_masked)]
            c2_vals = c2_masked[~np.isnan(c2_masked)]
            ax.scatter(self.params[0] * c2_vals + self.params[1], c1_vals, s=s)

        # Plot line
        ax.plot([0, max(self.gfp_vals)], [0, max(self.gfp_vals)], c='k', linestyle='--')

        # Finalise figure
        ax.set_xlim(np.percentile(self.gfp_vals, 0.01), np.percentile(self.gfp_vals, 99.99))
        ax.set_ylim(np.percentile(self.gfp_vals, 0.01), np.percentile(self.gfp_vals, 99.99))
        ax.set_xlabel('%.3f * AF + %.1f' % (self.params[0], self.params[1]))
        ax.set_ylabel('GFP')
        return fig, ax

    def _plot_prediction_3channel(self, s: float = 0.001):
        fig, ax = plt.subplots()

        # Scatter plot
        for c1, c2, c3, m in zip(self.gfp_filtered, self.af_filtered, self.rfp_filtered, self.mask):
            c1_masked = c1 * m
            c2_masked = c2 * m
            c3_masked = c3 * m
            c1_vals = c1_masked[~np.isnan(c1_masked)]
            c2_vals = c2_masked[~np.isnan(c2_masked)]
            c3_vals = c3_masked[~np.isnan(c3_masked)]
            ax.scatter(self.params[0] * c2_vals + self.params[1] * c3_vals + self.params[2], c1_vals, s=s)

        # Plot line
        ax.plot([0, max(self.gfp_vals)], [0, max(self.gfp_vals)], c='k', linestyle='--')

        # Finalise figure
        ax.set_xlim(np.percentile(self.gfp_vals, 0.01), np.percentile(self.gfp_vals, 99.99))
        ax.set_ylim(np.percentile(self.gfp_vals, 0.01), np.percentile(self.gfp_vals, 99.99))
        ax.set_xlabel('%.3f * AF + %.3f * RFP + %.1f' % (self.params[0], self.params[1], self.params[2]))
        ax.set_ylabel('GFP')
        return fig, ax

    def plot_residuals(self, s: float = 0.001):
        if self.rfp is None:
            fig, ax = self._plot_residuals_2channel(s=s)
        else:
            fig, ax = self._plot_residuals_3channel(s=s)
        return fig, ax

    def _plot_residuals_2channel(self, s: float = 0.001):
        fig, ax = plt.subplots()

        # Scatter
        for c1, c2, m in zip(self.gfp_filtered, self.af_filtered, self.mask):
            c1_masked = c1 * m
            c2_masked = c2 * m
            c1_vals = c1_masked[~np.isnan(c1_masked)]
            c2_vals = c2_masked[~np.isnan(c2_masked)]

            x = self.params[0] * c2_vals + self.params[1]
            resids = c1_vals - x
            ax.scatter(x, resids, s=s)

        # Plot line
        ax.axhline(0, linestyle='--', c='k')

        # Finalise figure
        ax.set_xlabel('%.3f * AF + %.1f' % (self.params[0], self.params[1]))
        ax.set_ylabel('Residuals')
        ax.set_xlim(np.percentile(self.gfp_vals, 0.01), np.percentile(self.gfp_vals, 99.99))
        return fig, ax

    def _plot_residuals_3channel(self, s: float = 0.001):
        fig, ax = plt.subplots()

        # Scatter plot
        for c1, c2, c3, m in zip(self.gfp_filtered, self.af_filtered, self.rfp_filtered, self.mask):
            c1_masked = c1 * m
            c2_masked = c2 * m
            c3_masked = c3 * m
            c1_vals = c1_masked[~np.isnan(c1_masked)]
            c2_vals = c2_masked[~np.isnan(c2_masked)]
            c3_vals = c3_masked[~np.isnan(c3_masked)]

            x = self.params[0] * c2_vals + self.params[1] * c3_vals + self.params[2]
            resids = c1_vals - x
            ax.scatter(x, resids, s=s)

        # Plot line
        ax.axhline(0, linestyle='--', c='k')

        # Finalise plot
        ax.set_xlabel('%.3f * AF + %.3f * RFP + %.1f' % (self.params[0], self.params[1], self.params[2]))
        ax.set_ylabel('Residuals')
        ax.set_xlim(np.percentile(self.gfp_vals, 0.01), np.percentile(self.gfp_vals, 99.99))
        return fig, ax


def af_correlation(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None, intercept0: bool = False,
                   method: str = 'OLS') -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Calculates pixel-by-pixel correlation between two channels
    Takes 3d image stacks shape [n, 512, 512]

    Args:
        img1: gfp channel
        img2: af channel
        mask: mask specifying region of images to use for regression, from make_mask function
        intercept0: if True, force the intercept to go through zero. Not recommended
        method: fitting method, either 'OLS' for ordinary least squares or 'ODR' for orthogonal distance regression

    Returns:
        parameters from regression [m, c]
        numpy array of all af channel pixels used for regression
        numpy array of all gfp channel pixels used for regression

    """

    # Convert to arrays
    if type(img1) is list:
        img1 = np.array(img1)
    if type(img2) is list:
        img2 = np.array(img2)
    if type(mask) is list:
        mask = np.array(mask)

    # Mask
    if mask is not None:
        img1 = img1 * mask
        img2 = img2 * mask

    # Flatten
    xdata = img2.flatten()
    ydata = img1.flatten()

    # Remove nans
    xdata = xdata[~np.isnan(xdata)]
    ydata = ydata[~np.isnan(ydata)]

    # Ordinary least squares regression
    if method == 'OLS':
        if not intercept0:
            lr = LinearRegression(fit_intercept=True)
            lr.fit(xdata[:, np.newaxis], ydata)
            params = [lr.coef_[0], lr.intercept_]

        else:
            lr = LinearRegression(fit_intercept=False)
            lr.fit(xdata[:, np.newaxis], ydata)
            params = [lr.coef_[0], 0]

    # Orthogonal distance regression
    elif method == 'ODR':
        if not intercept0:
            odr_mod = odr.Model(lambda b, x: b[0] * x + b[1])
            odr_data = odr.Data(xdata, ydata)
            odr_odr = odr.ODR(odr_data, odr_mod, beta0=[1, 0])
            output = odr_odr.run()
            params = [output.beta[0], output.beta[1]]
        else:
            odr_mod = odr.Model(lambda b, x: b[0] * x)
            odr_data = odr.Data(xdata, ydata)
            odr_odr = odr.ODR(odr_data, odr_mod, beta0=[1])
            output = odr_odr.run()
            params = [output.beta[0], 0]
    else:
        raise Exception('Method must be OLS or ODR')

    return params, xdata, ydata


def af_correlation_3channel(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray, mask: Optional[np.ndarray] = None,
                            intercept0: bool = False, method: str = 'OLS') -> Tuple[
    list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates pixel-by-pixel correlation between three channels
    Takes 3d image stacks shape [n, 512, 512]

    Args:
        img1: gfp channel
        img2: af channel
        img3: third channel
        mask: mask specifying region of images to use for regression, from make_mask function
        intercept0: if True, force the intercept to go through zero. Not recommended
        method: fitting method, either 'OLS' for ordinary least squares or 'ODR' for orthogonal distance regression

    Returns:
        parameters from regression [m, c]
        numpy array of all af channel pixels used for regression
        numpy array of all third channel pixels used for regression
        numpy array of all gfp channel pixels used for regression

    """

    # Convert to arrays
    if type(img1) is list:
        img1 = np.array(img1)
    if type(img2) is list:
        img2 = np.array(img2)
    if type(img3) is list:
        img3 = np.array(img3)
    if type(mask) is list:
        mask = np.array(mask)

    # Mask
    if mask is not None:
        img1 *= mask
        img2 *= mask
        img3 *= mask

    # Flatten
    xdata = img2.flatten()
    ydata = img3.flatten()
    zdata = img1.flatten()

    # Remove nans
    xdata = xdata[~np.isnan(xdata)]
    ydata = ydata[~np.isnan(ydata)]
    zdata = zdata[~np.isnan(zdata)]

    # Ordinary least squares regression
    if method == 'OLS':
        if not intercept0:
            lr = LinearRegression(fit_intercept=True)
            lr.fit(np.vstack((xdata, ydata)).T, zdata)
            params = [lr.coef_[0], lr.coef_[1], lr.intercept_]

        else:
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack((xdata, ydata)).T, zdata)
            params = [lr.coef_[0], lr.coef_[1], 0]

    # Orthogonal distance regression
    elif method == 'ODR':
        if not intercept0:
            odr_mod = odr.Model(lambda b, x: b[0] * x[0] + b[1] * x[1] + b[2])
            odr_data = odr.Data(np.c_[xdata, ydata].T, zdata)
            odr_odr = odr.ODR(odr_data, odr_mod, beta0=[1, 1, 0])
            output = odr_odr.run()
            params = [output.beta[0], output.beta[1], output.beta[2]]
        else:
            odr_mod = odr.Model(lambda b, x: b[0] * x[0] + b[1] * x[1])
            odr_data = odr.Data(np.c_[xdata, ydata].T, zdata)
            odr_odr = odr.ODR(odr_data, odr_mod, beta0=[1, 1])
            output = odr_odr.run()
            params = [output.beta[0], output.beta[1], 0]
    else:
        raise Exception('Method must be OLS or ODR')

    return params, xdata, ydata, zdata


def af_subtraction(ch1: np.ndarray, ch2: np.ndarray, m: float, c: float) -> np.ndarray:
    """
    Subtract ch2 from ch1
    ch2 is first adjusted to m * ch2 + c

    Args:
        ch1: numpy array of channel 1
        ch2: numpy array of channel 2
        m: m parameter obtained from n2 data
        c: c parameter obtained from n2 data

    Returns:
        numpy array of af corrected image

    """

    af = m * ch2 + c
    signal = ch1 - af
    return signal


def af_subtraction_3channel(ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray, m1: float, m2: float,
                            c: float) -> np.ndarray:
    """
    Subtract ch2 and ch3 from ch1
    ch2 and ch3 are first adjusted to m1 * ch2 + m2 * ch3 + c

    Args:
        ch1: numpy array of channel 1
        ch2: numpy array of channel 2
        ch3: numpy array of channel 3
        m1: m1 parameter obtained from n2 data
        m2: m2 parameter obtained from n2 data
        c: c parameter obtained from n2 data

    Returns:
        numpy array of af corrected image

    """

    af = m1 * ch2 + m2 * ch3 + c
    signal = ch1 - af
    return signal
