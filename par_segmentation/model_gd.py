import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from tqdm import tqdm

from ._tgf_interpolate import interpolate
from .funcs import (
    interp_1d_array,
    interp_2d_array,
    rolling_ave_2d,
    rotate_roi,
    straighten,
)
from .roi import interp_roi, offset_coordinates
from .model_base import ImageQuantBase

"""
TODO:
- use a spline based method for nfits

"""


class ImageQuantGradientDescent(ImageQuantBase):
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
        zerocap: bool = False,
        save_training: bool = False,
        save_sims: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            img=img,
            roi=roi,
        )

        # Model parameters
        self.periodic = periodic
        self.thickness = thickness
        self.rol_ave = rol_ave
        self.rotate = rotate
        self.nfits = nfits
        self.zerocap = zerocap
        self.roi_knots = roi_knots
        self.iterations = iterations
        self.rol_ave = rol_ave
        self.rotate = rotate
        self.sigma = sigma
        self.lr = lr
        self.descent_steps = descent_steps
        self.freedom = freedom
        self.fit_outer = fit_outer
        self.swish_factor = 10
        self.batch_norm = batch_norm
        self.adaptive_sigma = adaptive_sigma

        # Misc
        self.save_training = save_training
        self.save_sims = save_sims
        self.verbose = verbose

        # Tensors
        self.cyts_t = None
        self.mems_t = None
        self.offsets_t = None

        # Interpolated results
        self.mems_full = None
        self.cyts_full = None
        self.offsets_full = None

    """
    Run

    """

    def run(self):
        t = time.time()

        # Fitting
        for i in range(self.iterations):
            if self.verbose:
                print(f"Iteration {i + 1} of {self.iterations}")
            time.sleep(0.1)

            if i > 0:
                self._adjust_roi()
            self._fit()

        if self.verbose:
            time.sleep(0.1)
            print("Time elapsed: %.2f seconds \n" % (time.time() - t))

    def _preprocess(
        self, frame: np.ndarray, roi: np.ndarray
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Preprocesses a single image with roi specified

        Steps:
        - Straighten according to ROI
        - Apply rolling average
        - Either interpolated to a common length (self.nfits) or pad to length of
            largest image if nfits is not speficied
        - Normalise images, either to themselves or globally

        """

        # Straighten
        straight = straighten(
            frame, roi, thickness=self.thickness, interp="cubic", periodic=self.periodic
        )

        # Smoothen (rolling average)
        straight = rolling_ave_2d(straight, window=self.rol_ave, periodic=self.periodic)

        # Interpolate to a length nfits or pad smaller images to size of largest image
        if self.nfits is not None:
            straight = interp_2d_array(straight, self.nfits, ax=1, method="cubic")
            mask = np.ones(self.nfits)
        else:
            pad_size = max(r.shape[0] for r in self.roi)
            straight = np.pad(
                straight, pad_width=((0, 0), (0, (pad_size - straight.shape[1])))
            )
            mask = np.zeros(pad_size)
            mask[: straight.shape[1]] = 1

        # Normalise
        norm = np.percentile(straight, 99) if not self.batch_norm else 1
        straight /= norm

        return straight, norm, mask

    def _init_tensors(self):
        """
        Initialising offsets, cytoplasmic concentrations and membrane concentrations as zero
        Sigma initialised as user-specified value (or default), and may be trained
        """

        nimages = self.target.shape[0]
        self.vars = {}

        # Offsets
        self.offsets_t = tf.Variable(
            np.zeros([nimages, self.roi_knots]), name="Offsets"
        )
        if self.freedom != 0:
            self.vars["offsets"] = self.offsets_t

        # Cytoplasmic concentrations
        self.cyts_t = tf.Variable(
            np.zeros_like(np.mean(self.target[:, -5:, :], axis=1))
        )
        self.vars["cyts"] = self.cyts_t

        # Membrane concentrations
        self.mems_t = tf.Variable(np.zeros_like(np.max(self.target, axis=1)))
        self.vars["mems"] = self.mems_t

        # Outers
        if self.fit_outer:
            self.outers_t = tf.Variable(
                np.zeros_like(np.mean(self.target[:, :5, :], axis=1))
            )
            self.vars["outers"] = self.outers_t

        # Sigma
        self.sigma_t = tf.Variable(self.sigma, dtype=tf.float64)
        if self.adaptive_sigma:
            self.vars["sigma"] = self.sigma_t

    def _sim_images(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Simulates images according to current membrane and cytoplasm concentration
        estimates and offsets
        """

        nimages = self.mems_t.shape[0]
        nfits = (
            max([len(r[:, 0]) for r in self.roi]) if self.nfits is None else self.nfits
        )

        # Constrain concentrations
        mems = (
            self.mems_t * tf.math.sigmoid(self.swish_factor * self.mems_t)
            if self.zerocap
            else self.mems_t
        )
        cyts = (
            self.cyts_t * tf.math.sigmoid(self.swish_factor * self.cyts_t)
            if self.zerocap
            else self.cyts_t
        )

        # Create offsets spline and constrain offsets
        offsets_spline = create_offsets_spline(
            self.offsets_t, self.roi_knots, self.periodic, self.n, self.nfits, self.roi
        )
        offsets = self.freedom * tf.math.tanh(offsets_spline)

        # Positions to evaluate mem and cyt curves
        positions_ = tf.range(self.thickness, dtype=tf.float64)[
            tf.newaxis, tf.newaxis, :
        ]
        positions = tf.reshape(tf.math.add(positions_, offsets[:, :, tf.newaxis]), [-1])

        # Cap positions off edge
        positions = tf.clip_by_value(positions, 0, self.thickness - 1.000001)

        # Mask
        mask_ = tf.reshape(
            1
            - (
                tf.cast(tf.math.less(positions, 0), tf.float64)
                + tf.cast(tf.math.greater(positions, self.thickness), tf.float64)
            ),
            [nimages, nfits, self.thickness],
        )

        # Mem curve
        mem_curve = tf.reshape(
            tf.math.exp(
                -((positions - self.thickness / 2) ** 2) / (2 * self.sigma_t**2)
            ),
            [nimages, nfits, self.thickness],
        )

        # Cyt curve
        cyt_curve = tf.reshape(
            (
                1
                + tf.math.erf(
                    (positions - self.thickness / 2) / (self.sigma_t * (2**0.5))
                )
            )
            / 2,
            [nimages, nfits, self.thickness],
        )

        # Calculate output
        mem_total = mem_curve * tf.expand_dims(mems, axis=-1)
        cyt_total = (
            tf.expand_dims(self.outers_t, axis=-1)
            + cyt_curve * tf.expand_dims((cyts - self.outers_t), axis=-1)
            if self.fit_outer
            else cyt_curve * tf.expand_dims(cyts, axis=-1)
        )

        # Sum outputs
        return tf.transpose(tf.math.add(mem_total, cyt_total), [0, 2, 1]), tf.transpose(
            mask_, [0, 2, 1]
        )

    def _losses_full(self) -> tf.Tensor:
        """
        Calculates the mean squared error (MSE) loss between the simulated and target
        images.
        """

        # Simulate images
        self.sim, mask = self._sim_images()

        # Calculate squared errors
        sq_errors = tf.square(self.sim - self.target)

        # Apply mask if different size images are used
        if self.nfits is None:
            mask *= tf.expand_dims(self.masks, axis=1)

        # Calculate masked average
        mse = tf.reduce_sum(sq_errors * mask, axis=[1, 2]) / tf.reduce_sum(
            mask, axis=[1, 2]
        )

        return mse

    def _fit(self):
        # Preprocess
        target, norms, masks = zip(
            *[self._preprocess(frame, roi) for frame, roi in zip(self.img, self.roi)]
        )
        self.target = np.array(target)
        self.norms = np.array(norms)
        self.masks = np.array(masks)

        # Batch normalise
        if self.batch_norm:
            norm = np.percentile(self.target, 99)
            self.target /= norm
            self.norms = np.ones(self.target.shape[0]) * norm

        # Init tensors
        self._init_tensors()

        # Run optimisation
        self.saved_vars = []
        self.saved_sims = []
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.losses = np.zeros([len(self.img), self.descent_steps])

        iterable = (
            tqdm(range(self.descent_steps))
            if self.verbose
            else range(self.descent_steps)
        )

        for i in iterable:
            with tf.GradientTape() as tape:
                losses_full = self._losses_full()
                self.losses[:, i] = losses_full
                loss = tf.reduce_mean(losses_full)
                grads = tape.gradient(loss, self.vars.values())
                opt.apply_gradients(zip(grads, self.vars.values()))

            # Save trained variables
            if self.save_training:
                self.saved_vars.append(
                    {key: value.numpy() for key, value in self.vars.items()}
                )

            # Save interim simulations
            if self.save_sims:
                self.saved_sims.append(
                    self.sim.numpy() * self.norms[:, np.newaxis, np.newaxis]
                )

        # Save and rescale sim images (rescaled)
        self.sim_both, self.target = (
            data * self.norms[:, np.newaxis, np.newaxis]
            for data in [self._sim_images()[0].numpy(), self.target]
        )

        # Save and rescale results
        mems, cyts = (
            data * tf.math.sigmoid(self.swish_factor * data) if self.zerocap else data
            for data in [self.mems_t, self.cyts_t]
        )

        self.mems, self.cyts = (
            data.numpy() * self.norms[:, np.newaxis] for data in [mems, cyts]
        )

        # Create offsets spline
        offsets_spline = create_offsets_spline(
            self.offsets_t, self.roi_knots, self.periodic, self.n, self.nfits, self.roi
        )

        # Constrain offsets
        self.offsets = self.freedom * tf.math.tanh(offsets_spline)

        # Crop results
        if self.nfits is None:
            self.offsets, self.cyts, self.mems = (
                [data[mask == 1] for data, mask in zip(dataset, self.masks)]
                for dataset in [self.offsets, self.cyts, self.mems]
            )

        # Interpolated results
        if self.nfits is not None:
            self.offsets_full, self.cyts_full, self.mems_full = (
                [
                    interp_1d_array(data, len(roi[:, 0]), method=method)
                    for data, roi in zip(dataset, self.roi)
                ]
                for dataset, method in zip(
                    [self.offsets, self.cyts, self.mems], ["cubic", "linear", "linear"]
                )
            )
        else:
            self.offsets_full, self.cyts_full, self.mems_full = (
                self.offsets,
                self.cyts,
                self.mems,
            )

        # Interpolated sim images
        if self.nfits is not None:
            self.straight_images_sim, self.straight_images = (
                [
                    interp1d(np.arange(self.nfits), data, axis=-1)(
                        np.linspace(0, self.nfits - 1, len(roi[:, 0]))
                    )
                    for roi, data in zip(self.roi, dataset)
                ]
                for dataset in [self.sim_both, self.target]
            )
        else:
            self.straight_images_sim, self.straight_images = (
                [data.T[mask == 1].T for data, mask in zip(dataset, self.masks)]
                for dataset in [self.sim_both, self.target]
            )

        self.straight_images_resids = [
            i - j for i, j in zip(self.straight_images, self.straight_images_sim)
        ]

        # Save adaptable params
        if self.sigma is not None:
            self.sigma = self.sigma_t.numpy()

    """
    Misc

    """

    def _adjust_roi(self):
        """
        Adjusts the region of interest (ROI) after a preliminary fit to refine coordinates.
        A refit must be performed after this adjustment.
        """

        # Offset coordinates and interpolate ROI
        self.roi = [
            interp_roi(offset_coordinates(roi, offsets), periodic=self.periodic)
            for roi, offsets in zip(self.roi, self.offsets_full)
        ]

        # Rotate ROI if periodic and rotation is enabled
        if self.periodic and self.rotate:
            self.roi = [rotate_roi(roi) for roi in self.roi]

    """
    Interactive
    
    """

    def plot_losses(self, log: bool = False):
        fig, ax = plt.subplots()
        losses = np.log10(self.losses.T) if log else self.losses.T
        ax.plot(losses)
        ax.set_xlabel("Descent step")
        ax.set_ylabel("log10(Mean square error)" if log else "Mean square error")
        return fig, ax


def create_offsets_spline(
    offsets_t, roi_knots, periodic, nimages, nfits, roi
) -> tf.Tensor:
    # Determine nfits if not provided
    if nfits is None:
        nfits = max(len(r[:, 0]) for r in roi)

    # Create offsets spline
    x = np.tile(
        np.expand_dims(
            np.arange(-1.0, roi_knots + 2 if periodic else roi_knots + 1), 0
        ),
        (nimages, 1),
    )
    y = tf.concat(
        (
            offsets_t[:, -1:] if periodic else offsets_t[:, :1],
            offsets_t,
            offsets_t[:, :2] if periodic else offsets_t[:, -1:],
        ),
        axis=1,
    )
    knots = tf.stack((x, y))

    # Evaluate offset spline
    positions = tf.expand_dims(
        tf.cast(
            tf.linspace(
                start=0.0,
                stop=roi_knots if periodic else roi_knots - 1.000001,
                num=nfits + 1 if periodic else nfits,
            )[: -1 if periodic else None],
            dtype=tf.float64,
        ),
        axis=-1,
    )

    if nfits is not None:
        spline = interpolate(knots, positions, degree=3, cyclical=False)
        offsets_spline = tf.transpose(tf.squeeze(spline, axis=1)[:, 1, :])
    else:
        offsets_spline = []
        for i in tf.range(nimages):
            positions = tf.expand_dims(
                tf.cast(
                    tf.linspace(
                        start=0.0,
                        stop=roi_knots if periodic else roi_knots - 1.000001,
                        num=roi[i].shape[0] + 1 if periodic else roi[i].shape[0],
                    )[: -1 if periodic else None],
                    dtype=tf.float64,
                ),
                axis=-1,
            )
            spline = interpolate(
                knots[:, i : i + 1, :], positions, degree=3, cyclical=False
            )
            spline = tf.transpose(tf.squeeze(spline, axis=1)[:, 1, :])[0]
            pad = tf.zeros([nfits - roi[i].shape[0]], dtype=tf.float64)
            offsets_spline.append(tf.concat([spline, pad], axis=0))
        offsets_spline = tf.stack(offsets_spline, axis=0)

    return offsets_spline
