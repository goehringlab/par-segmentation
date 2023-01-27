import numpy as np
import tensorflow as tf
from .funcs import straighten, rolling_ave_2d, interp_1d_array, interp_2d_array, rotate_roi
from .roi import offset_coordinates, interp_roi
from scipy.interpolate import interp1d
from tqdm import tqdm
import time
from .tgf_interpolate import interpolate
import matplotlib.pyplot as plt
from typing import Union, Tuple


class ImageQuantGradientDescent:

    def __init__(self,
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
                 descent_steps: int = 300,
                 adaptive_sigma: bool = False,
                 batch_norm: bool = False,
                 freedom: float = 10,
                 roi_knots: int = 20,
                 fit_outer: bool = True,
                 zerocap: bool = False,
                 save_training: bool = False,
                 save_sims: bool = False,
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

        self.periodic = periodic
        self.roi_knots = roi_knots

        # Normalisation
        self.batch_norm = batch_norm

        # Fitting parameters
        self.iterations = iterations
        self.thickness = thickness
        self.rol_ave = rol_ave
        self.rotate = rotate
        self.sigma = sigma
        self.nfits = nfits
        self.lr = lr
        self.descent_steps = descent_steps
        self.freedom = freedom
        self.fit_outer = fit_outer
        self.save_training = save_training
        self.save_sims = save_sims
        self.zerocap = zerocap
        self.swish_factor = 10
        self.verbose = verbose

        # Learning
        self.adaptive_sigma = adaptive_sigma

        # Results containers
        self.offsets = None
        self.cyts = None
        self.mems = None
        self.offsets_full = None
        self.cyts_full = None
        self.mems_full = None

        # Tensors
        self.cyts_t = None
        self.mems_t = None
        self.offsets_t = None

    """
    Run

    """

    def run(self):
        t = time.time()

        # Fitting
        for i in range(self.iterations):
            if self.verbose:
                print(f'Iteration {i + 1} of {self.iterations}')
            time.sleep(0.1)

            if i > 0:
                self.adjust_roi()
            self.fit()

        if self.verbose:
            time.sleep(0.1)
            print('Time elapsed: %.2f seconds \n' % (time.time() - t))

    def preprocess(self, frame: np.ndarray, roi: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Preprocesses a single image with roi specified

        Steps:
        - Straighten according to ROI
        - Apply rolling average
        - Either interpolated to a common length (self.nfits) or pad to length of largest image if nfits is not speficied
        - Normalise images, either to themselves or globally

        """

        # Straighten
        straight = straighten(frame, roi, thickness=self.thickness, interp='cubic', periodic=self.periodic)

        # Smoothen (rolling average)
        straight = rolling_ave_2d(straight, window=self.rol_ave, periodic=self.periodic)

        # Interpolate to a length nfits
        if self.nfits is not None:
            straight = interp_2d_array(straight, self.nfits, ax=1, method='cubic')

        # If nfits not specified, pad smaller images to size of largest image
        if self.nfits is None:
            pad_size = max([r.shape[0] for r in self.roi])
            target = np.pad(straight, pad_width=((0, 0), (0, (pad_size - straight.shape[1]))))
            mask = np.zeros(pad_size)
            mask[:straight.shape[1]] = 1
        else:
            mask = np.ones(self.nfits)
            target = straight

        # Normalise
        if not self.batch_norm:
            norm = np.percentile(straight, 99)
            target /= norm
        else:
            norm = 1

        return target, norm, mask

    def init_tensors(self):
        """
        Initialising offsets, cytoplasmic concentrations and membrane concentrations as zero
        Sigma initialised as user-specified value (or default), and may be trained

        """

        nimages = self.target.shape[0]
        self.vars = {}

        # Offsets
        self.offsets_t = tf.Variable(np.zeros([nimages, self.roi_knots]), name='Offsets')
        if not self.freedom == 0:
            self.vars['offsets'] = self.offsets_t

        # Cytoplasmic concentrations
        self.cyts_t = tf.Variable(0 * np.mean(self.target[:, -5:, :], axis=1))
        self.vars['cyts'] = self.cyts_t

        # Membrane concentrations
        self.mems_t = tf.Variable(0 * np.max(self.target, axis=1))
        self.vars['mems'] = self.mems_t

        # Outers
        if self.fit_outer:
            self.outers_t = tf.Variable(0 * np.mean(self.target[:, :5, :], axis=1))
            self.vars['outers'] = self.outers_t

        # Sigma
        self.sigma_t = tf.Variable(self.sigma, dtype=tf.float64)
        if self.adaptive_sigma:
            self.vars['sigma'] = self.sigma_t

    def sim_images(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Simulates images according to current membrane and cytoplasm concentration estimates and offsets

        """

        nimages = self.mems_t.shape[0]

        # Specify number of fits
        if self.nfits is None:
            nfits = max([len(r[:, 0]) for r in self.roi])
        else:
            nfits = self.nfits

        # Constrain concentrations
        mems = self.mems_t
        cyts = self.cyts_t
        if self.zerocap:
            mems = mems * tf.math.sigmoid(self.swish_factor * mems)
            cyts = cyts * tf.math.sigmoid(self.swish_factor * cyts)

        # Create offsets spline
        offsets_spline = create_offsets_spline(self.offsets_t, self.roi_knots, self.periodic, self.n, self.nfits,
                                               self.roi)

        # Constrain offsets
        offsets = self.freedom * tf.math.tanh(offsets_spline)

        # Positions to evaluate mem and cyt curves
        positions_ = np.arange(self.thickness, dtype=np.float64)[tf.newaxis, tf.newaxis, :]
        offsets_ = offsets[:, :, tf.newaxis]
        positions = tf.reshape(tf.math.add(positions_, offsets_), [-1])

        # Cap positions off edge
        positions = tf.minimum(positions, self.thickness - 1.000001)
        positions = tf.maximum(positions, 0)

        # Mask
        mask = 1 - (tf.cast(tf.math.less(positions, 0), tf.float64) + tf.cast(
            tf.math.greater(positions, self.thickness), tf.float64))
        mask_ = tf.reshape(mask, [nimages, nfits, self.thickness])

        # Mem curve
        mem_curve = tf.math.exp(-((positions - self.thickness / 2) ** 2) / (2 * self.sigma_t ** 2))
        mem_curve = tf.reshape(mem_curve, [nimages, nfits, self.thickness])

        # Cyt curve
        cyt_curve = (1 + tf.math.erf((positions - self.thickness / 2) / (self.sigma_t * (2 ** 0.5)))) / 2
        # self.sigma_t * (2 ** 0.5) <- this is what happens when a step is convolved with a Gaussian
        cyt_curve = tf.reshape(cyt_curve, [nimages, nfits, self.thickness])

        # Calculate output
        mem_total = mem_curve * tf.expand_dims(mems, axis=-1)
        if not self.fit_outer:
            cyt_total = cyt_curve * tf.expand_dims(cyts, axis=-1)
        else:
            cyt_total = tf.expand_dims(self.outers_t, axis=-1) + cyt_curve * tf.expand_dims((cyts - self.outers_t),
                                                                                            axis=-1)
        # Sum outputs
        return tf.transpose(tf.math.add(mem_total, cyt_total), [0, 2, 1]), tf.transpose(mask_, [0, 2, 1])

    def losses_full(self) -> tf.Tensor:

        # Simulate images
        self.sim, mask = self.sim_images()

        # Calculate errors
        sq_errors = (self.sim - self.target) ** 2

        # Masking (when different size images are used)
        if self.nfits is None:
            mask *= tf.expand_dims(self.masks, axis=1)

        # Masked average
        mse = tf.reduce_sum(sq_errors * mask, axis=[1, 2]) / tf.reduce_sum(mask, axis=[1, 2])
        return mse

    def fit(self):

        # Preprocess
        target, norms, masks = zip(*[self.preprocess(frame, roi) for frame, roi in zip(self.img, self.roi)])
        self.target = np.array(target)
        self.norms = np.array(norms)
        self.masks = np.array(masks)

        # Batch normalise
        if self.batch_norm:
            norm = np.percentile(self.target, 99)
            self.target /= norm
            self.norms = norm * np.ones(self.target.shape[0])

        # Init tensors
        self.init_tensors()

        # Run optimisation
        self.saved_vars = []
        self.saved_sims = []
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.losses = np.zeros([len(self.img), self.descent_steps])
        if self.verbose:
            iterable = tqdm(range(self.descent_steps))
        else:
            iterable = range(self.descent_steps)
        for i in iterable:
            with tf.GradientTape() as tape:
                losses_full = self.losses_full()
                self.losses[:, i] = losses_full
                loss = tf.reduce_mean(losses_full)
                grads = tape.gradient(loss, self.vars.values())
                opt.apply_gradients(list(zip(grads, self.vars.values())))

            # Save trained variables
            if self.save_training:
                newdict = {key: value.numpy() for key, value in self.vars.items()}
                self.saved_vars.append(newdict)

            # Save interim simulations
            if self.save_sims:
                self.saved_sims.append(self.sim.numpy() * self.norms[:, np.newaxis, np.newaxis])

        # Save and rescale sim images (rescaled)
        self.sim_both = self.sim_images()[0].numpy() * self.norms[:, np.newaxis, np.newaxis]
        self.target = self.target * self.norms[:, np.newaxis, np.newaxis]

        # Save and rescale results
        mems = self.mems_t
        cyts = self.cyts_t
        if self.zerocap:
            mems = mems * tf.math.sigmoid(self.swish_factor * mems)
            cyts = cyts * tf.math.sigmoid(self.swish_factor * cyts)
        self.mems = mems.numpy() * self.norms[:, np.newaxis]
        self.cyts = cyts.numpy() * self.norms[:, np.newaxis]

        # Create offsets spline
        offsets_spline = create_offsets_spline(self.offsets_t, self.roi_knots, self.periodic, self.n, self.nfits,
                                               self.roi)

        # Constrain offsets
        self.offsets = self.freedom * tf.math.tanh(offsets_spline)

        # Crop results
        if self.nfits is None:
            self.offsets = [offsets[mask == 1] for offsets, mask in zip(self.offsets, self.masks)]
            self.cyts = [cyts[mask == 1] for cyts, mask in zip(self.cyts, self.masks)]
            self.mems = [mems[mask == 1] for mems, mask in zip(self.mems, self.masks)]

        # Interpolated results
        if self.nfits is not None:
            self.offsets_full = [interp_1d_array(offsets, len(roi[:, 0]), method='cubic') for offsets, roi in
                                 zip(self.offsets, self.roi)]
            self.cyts_full = [interp_1d_array(cyts, len(roi[:, 0]), method='linear') for cyts, roi in
                              zip(self.cyts, self.roi)]
            self.mems_full = [interp_1d_array(mems, len(roi[:, 0]), method='linear') for mems, roi in
                              zip(self.mems, self.roi)]
        else:
            self.offsets_full = self.offsets
            self.cyts_full = self.cyts
            self.mems_full = self.mems

        # Interpolated sim images
        if self.nfits is not None:
            self.sim_full = [interp1d(np.arange(self.nfits), sim_both, axis=-1)(
                np.linspace(0, self.nfits - 1, len(roi[:, 0]))) for roi, sim_both in
                zip(self.roi, self.sim_both)]
            self.target_full = [interp1d(np.arange(self.nfits), target, axis=-1)(
                np.linspace(0, self.nfits - 1, len(roi[:, 0]))) for roi, target in zip(self.roi, self.target)]
            self.resids_full = [i - j for i, j in zip(self.target_full, self.sim_full)]
        else:
            self.sim_full = [sim_both.T[mask == 1].T for sim_both, mask in zip(self.sim_both, self.masks)]
            self.target_full = [target.T[mask == 1].T for target, mask in zip(self.target, self.masks)]
            self.resids_full = [i - j for i, j in zip(self.target_full, self.sim_full)]

        # Save adaptable params
        if self.sigma is not None:
            self.sigma = self.sigma_t.numpy()

    """
    Misc

    """

    def adjust_roi(self):
        """
        Can do after a preliminary fit to refine coordinates
        Must refit after doing this

        """

        # Offset coordinates
        self.roi = [interp_roi(offset_coordinates(roi, offsets_full), periodic=self.periodic) for roi, offsets_full in
                    zip(self.roi, self.offsets_full)]

        # Rotate
        if self.periodic:
            if self.rotate:
                self.roi = [rotate_roi(roi) for roi in self.roi]

    """
    Interactive
    
    """

    def plot_losses(self, log: bool = False):
        fig, ax = plt.subplots()
        if not log:
            ax.plot(self.losses.T)
            ax.set_xlabel('Descent step')
            ax.set_ylabel('Mean square error')
        else:
            ax.plot(np.log10(self.losses.T))
            ax.set_xlabel('Descent step')
            ax.set_ylabel('log10(Mean square error)')
        return fig, ax


def create_offsets_spline(offsets_t, roi_knots, periodic, nimages, nfits, roi) -> tf.Tensor:
    if nfits is None:
        nfits = max([len(r[:, 0]) for r in roi])

    # Create offsets spline
    if periodic:
        x = np.tile(np.expand_dims(np.arange(-1., roi_knots + 2), 0), (nimages, 1))
        y = tf.concat((offsets_t[:, -1:], offsets_t, offsets_t[:, :2]), axis=1)
        knots = tf.stack((x, y))
    else:
        x = np.tile(np.expand_dims(np.arange(-1., roi_knots + 1), 0), (nimages, 1))
        y = tf.concat((offsets_t[:, :1], offsets_t, offsets_t[:, -1:]), axis=1)
        knots = tf.stack((x, y))

    # Evaluate offset spline
    if nfits is not None:
        if periodic:
            positions = tf.expand_dims(tf.cast(tf.linspace(start=0.0, stop=roi_knots,
                                                           num=nfits + 1)[:-1], dtype=tf.float64), axis=-1)
        else:
            positions = tf.expand_dims(tf.cast(tf.linspace(start=0.0, stop=roi_knots - 1.000001,
                                                           num=nfits), dtype=tf.float64), axis=-1)
        spline = interpolate(knots, positions, degree=3, cyclical=False)
        spline = tf.squeeze(spline, axis=1)
        offsets_spline = tf.transpose(spline[:, 1, :])

    else:
        offsets_spline = []
        for i in tf.range(nimages):
            if periodic:
                positions = tf.expand_dims(
                    tf.cast(tf.linspace(start=0.0, stop=roi_knots, num=roi[i].shape[0] + 1)[:-1],
                            dtype=tf.float64), axis=-1)
            else:
                positions = tf.expand_dims(tf.cast(
                    tf.linspace(start=0.0, stop=roi_knots - 1.000001, num=roi[i].shape[0]),
                    dtype=tf.float64), axis=-1)
            spline = interpolate(knots[:, i:i + 1, :], positions, degree=3, cyclical=False)
            spline = tf.squeeze(spline, axis=1)
            spline = tf.transpose(spline[:, 1, :])[0]
            pad = tf.zeros([nfits - roi[i].shape[0]], dtype=tf.float64)
            offsets_spline.append(tf.concat([spline, pad], axis=0))
        offsets_spline = tf.stack(offsets_spline, axis=0)

    return offsets_spline
