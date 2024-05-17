import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.nn import sigmoid
from scipy.special import erf
from skimage.measure import block_reduce
from tqdm import tqdm

from .funcs import interp_2d_array, rolling_ave_2d, straighten
from .roi import interp_roi, offset_coordinates
from .model_base import ImageQuantBase

"""
TODO:
- plot_fits image looks squashed when pooling rate != 1
- support for non-periodic ROIs
- options to control verbosity
- automatically terminate optimisation when stable

"""


class ImageQuantFlexi(ImageQuantBase):
    def __init__(
        self,
        img,
        roi,
        cytbg=None,
        membg=None,
        thickness=50,
        batch_norm=False,
        zerocap=True,
        pooling_rate=1,
        norm_factor=None,
        rol_ave=1,
        nfits=None,
    ):
        super().__init__(
            img=img,
            roi=roi,
        )

        # Core parameters
        self.thickness = thickness
        self.nfits = nfits

        # Image preprocessing
        self.rol_ave = rol_ave
        self.batch_norm = batch_norm
        self.norm_factor = norm_factor
        self.downsampling_rate = pooling_rate

        # Fitting parameters
        self.swish_factor = 30
        self.zerocap = zerocap

        # Membrane/cytoplasmic reference profile
        self.cytbg = cytbg
        self.membg = membg
        self.sigma = 1.5

        # Internal variables for simulations (numpy arrays and tensors)
        self.target = None
        self.sim = None
        self.norms = None
        self.masks = None
        self.losses = None
        self.mems_opt = None
        self.cyts_opt = None
        self.cytbg_opt = None
        self.membg_opt = None

        # Calculate padded size
        self.padded_size = int(
            max([np.ceil(r.shape[0] / pooling_rate) for r in self.roi])
        )

    """
    Run

    """

    def segment(self, freedom=10, roi_knots=20, lr=0.01, descent_steps=500):
        # Set up segmenter class
        from .quantifier import ImageQuant

        iq = ImageQuant(
            img=self.img,
            roi=self.roi,
            periodic=True,
            thickness=self.thickness,
            rol_ave=5,
            rotate=False,
            nfits=100,
            iterations=1,
            lr=lr,
            descent_steps=descent_steps,
            adaptive_sigma=False,
            batch_norm=False,
            freedom=freedom,
            roi_knots=roi_knots,
            fit_outer=True,
            save_training=False,
            save_sims=False,
            method="GD",
            itp=10,
            parallel=False,
            zerocap=False,
            cores=None,
            bg_subtract=False,
            interp="cubic",
            sigma=3.5,
        )

        # Run segmentation
        iq.run()

        # Save loss curves
        self.losses = iq.iq.losses

        # Offset coordinates and save
        self.roi[:] = [
            interp_roi(offset_coordinates(roi, offsets_full), periodic=True)
            for roi, offsets_full in zip(iq.roi, iq.offsets_full)
        ]
        self.padded_size = max([r.shape[0] for r in self.roi])

    def quantify(self, lr=0.005, descent_steps=600, save_interim=False):
        # Pre-process
        self._preprocess_batch()

        # Init tensors
        self._init_params()

        # Gradient descent
        res_interim = self._gradient_descent_quantification(
            lr=lr, descent_steps=descent_steps, save_interim=save_interim
        )

        # Store results
        self._store()

        # Return interim simulations
        if save_interim:
            return res_interim

    def calibrate_cytoplasm(self, lr=0.005, descent_steps=600, save_interim=False):
        # Pre-process
        self._preprocess_batch()

        # Init tensors
        self._init_params()

        # Cytbg calibration
        res_interim = self._gradient_descent_cytoplasm_calibration(
            lr=lr, descent_steps=descent_steps, save_interim=save_interim
        )

        # Store results
        self._store()

        # Return interim simulations
        if save_interim:
            return res_interim

    def calibrate_membrane(self, lr=0.005, descent_steps=600, save_interim=False):
        # Pre-process
        self._preprocess_batch()

        # Init tensors
        self._init_params()

        # Cytbg calibration
        res_interim = self._gradient_descent_membrane_calibration(
            lr=lr, descent_steps=descent_steps, save_interim=save_interim
        )

        # Store results
        self._store()

        # Return interim simulations
        if save_interim:
            return res_interim

    """
    Gradient descent
    
    """

    def _gradient_descent_quantification(self, lr, descent_steps, save_interim):
        # Initialisation
        opt = optax.adam(learning_rate=lr)  # Set up optimiser
        params = {
            "cyts_opt": self.cyts_opt,
            "mems_opt": self.mems_opt,
        }  # Create params list
        opt_state = opt.init(params)  # Set up optimiser initial state
        self.losses = np.zeros([self.n, descent_steps])  # Create empty losses array

        # Store initial values
        if save_interim:
            params_interim = {
                "cyts": [
                    params["cyts_opt"],
                ],
                "mems": [
                    params["mems_opt"],
                ],
            }

        # Define loss function
        @jax.jit
        def loss_function(_params):
            sim = sim_img_batch(
                _params["cyts_opt"],
                _params["mems_opt"],
                self.cytbg_opt,
                self.membg_opt,
                self.zerocap,
                self.swish_factor,
            )
            loss_full = masked_loss_function(sim, self.target, self.masks)
            return jnp.mean(loss_full), loss_full

        # Descent steps
        func_grad = jax.grad(loss_function, has_aux=True)
        for e in tqdm(range(descent_steps)):
            # Calculate gradients
            grads, losses_full = func_grad(params)
            self.losses[:, e] = losses_full

            # Scale gradients <- ensures training is invariant of batch size and pooling rate
            grads["cyts_opt"] *= self.n
            grads["mems_opt"] *= self.n * self.padded_size

            # Update parameters
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            # Save interim parameters
            if save_interim:
                params_interim["cyts"].append(params["cyts_opt"])
                params_interim["mems"].append(params["mems_opt"])

        # Save optimised parameters
        self.cyts_opt = params["cyts_opt"]
        self.mems_opt = params["mems_opt"]

        # Return interim parameters
        if save_interim:
            return params_interim
        else:
            return None

    def _gradient_descent_membrane_calibration(self, lr, descent_steps, save_interim):
        self.losses = np.zeros([self.n, descent_steps])  # Create empty losses array

        # Initialisation
        opt = optax.adam(learning_rate=lr)  # Set up optimiser
        params = {
            "cyts_opt": self.cyts_opt,
            "mems_opt": self.mems_opt,
            "membg_opt": self.membg_opt,
        }  # Create params list
        opt_state = opt.init(params)  # Set up optimiser initial state
        self.losses = np.zeros([self.n, descent_steps])  # Create empty losses array

        # Store initial values
        if save_interim:
            params_interim = {
                "cyts": [
                    params["cyts_opt"],
                ],
                "mems": [
                    params["mems_opt"],
                ],
                "membg": [
                    params["membg_opt"],
                ],
            }

        # Define loss function
        @jax.jit
        def loss_function(params):
            sim = sim_img_batch(
                params["cyts_opt"],
                params["mems_opt"],
                self.cytbg_opt,
                params["membg_opt"],
                self.zerocap,
                self.swish_factor,
            )
            loss_full = masked_loss_function(sim, self.target, self.masks)
            return jnp.mean(loss_full), loss_full

        # Descent steps
        func_grad = jax.grad(loss_function, has_aux=True)
        for e in tqdm(range(descent_steps)):
            # Calculate gradients
            grads, losses_full = func_grad(params)
            self.losses[:, e] = losses_full

            # Scale gradients
            grads["cyts_opt"] *= self.n
            grads["mems_opt"] *= self.n * self.padded_size

            # Update parameters
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            # Save interim parameters
            if save_interim:
                params_interim["cyts"].append(params["cyts_opt"])
                params_interim["mems"].append(params["mems_opt"])
                params_interim["membg"].append(params["membg_opt"])

        # Save optimised parameters
        self.cyts_opt = params["cyts_opt"]
        self.mems_opt = params["mems_opt"]
        self.membg_opt = params["membg_opt"]

        # Return interim parameters
        if save_interim:
            return params_interim
        else:
            return None

    def _gradient_descent_cytoplasm_calibration(self, lr, descent_steps, save_interim):
        # Initialisation
        opt = optax.adam(learning_rate=lr)  # Set up optimiser
        params = {
            "cyts_opt": self.cyts_opt,
            "cytbg_opt": self.cytbg_opt,
        }  # Create params list
        opt_state = opt.init(params)  # Set up optimiser initial state
        self.losses = np.zeros([self.n, descent_steps])  # Create empty losses array

        # Store initial values
        if save_interim:
            params_interim = {
                "cyts": [
                    params["cyts_opt"],
                ],
                "cytbg": [
                    params["cytbg_opt"],
                ],
            }

        # Define loss function
        @jax.jit
        def loss_function(_params):
            sim = sim_img_batch(
                _params["cyts_opt"],
                self.mems_opt,
                _params["cytbg_opt"],
                self.membg_opt,
                self.zerocap,
                self.swish_factor,
            )
            loss_full = masked_loss_function(sim, self.target, self.masks)
            return jnp.mean(loss_full), loss_full

        # Descent steps
        func_grad = jax.grad(loss_function, has_aux=True)
        for e in tqdm(range(descent_steps)):
            # Calculate gradients
            grads, losses_full = func_grad(params)
            self.losses[:, e] = losses_full

            # Scale gradients
            grads["cyts_opt"] *= self.n

            # Update parameters
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            # Save interim parameters
            if save_interim:
                params_interim["cyts"].append(params["cyts_opt"])
                params_interim["cytbg"].append(params["cytbg_opt"])

        # Save optimised parameters
        self.cyts_opt = params["cyts_opt"]
        self.cytbg_opt = params["cytbg_opt"]

        # Return interim parameters
        if save_interim:
            return params_interim
        else:
            return None

    """
    Preprocessing
    
    """

    def _preprocess_single(self, frame, roi):
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
            frame, roi, thickness=self.thickness, interp="cubic", periodic=True
        )

        # Smoothen (rolling average)
        if self.rol_ave > 1:
            straight = rolling_ave_2d(straight, window=self.rol_ave)

        # Interpolate
        if self.nfits is not None:
            straight = interp_2d_array(straight, self.nfits, ax=1, method="cubic")

        # Average pooling (may need to pad first) - could improve padding by using periodicity
        if self.downsampling_rate != 1:
            remainder = straight.shape[1] % self.downsampling_rate
            if remainder != 0:
                pad = np.repeat(
                    np.expand_dims(np.mean(straight[:, -remainder:], axis=1), -1),
                    (self.downsampling_rate - remainder),
                    axis=1,
                )
                straight = np.concatenate((straight, pad), axis=1)
            straight = block_reduce(straight, (1, self.downsampling_rate), np.mean)

        # Pad to size of largest image
        pad_size = self.padded_size - straight.shape[1]
        target = np.pad(straight, pad_width=((0, 0), (0, pad_size)))
        mask = np.zeros(self.padded_size)
        mask[: straight.shape[1]] = 1

        # Normalise
        if not self.batch_norm:
            norm = np.percentile(straight, 99)
            target /= norm
        else:
            norm = 1

        return target, norm, mask

    def _preprocess_batch(self):
        # Preprocess
        target, norms, masks = zip(
            *[
                self._preprocess_single(frame, roi)
                for frame, roi in zip(self.img, self.roi)
            ]
        )
        self.target = jnp.array(target)
        self.norms = jnp.array(norms)
        self.masks = jnp.array(masks)

        # Batch normalise
        if self.batch_norm:
            if self.norm_factor is not None:
                norm = self.norm_factor
            else:
                norm = np.percentile(self.target, 99)
            self.target /= norm
            self.norms = norm * np.ones(self.target.shape[0])

    """
    Simulation
    
    """

    def _init_params(self):
        # Cytoplasmic concentrations
        self.cyts_opt = 0 * np.mean(self.target[:, -5:, :], axis=(1, 2))

        # Membrane concentrations
        self.mems_opt = 0 * np.max(self.target, axis=1)

        # Cytoplasmic reference profile
        if self.cytbg is not None:
            self.cytbg_opt = self.cytbg
        else:
            # Initialise as error function
            self.cytbg_opt = (
                1 + erf((np.arange(self.thickness) - self.thickness / 2) / self.sigma)
            ) / 2

        # Membrane reference profile
        if self.membg is not None:
            self.membg_opt = self.membg
        else:
            # Initialise as Gaussian
            self.membg_opt = np.exp(
                -((np.arange(self.thickness) - self.thickness / 2) ** 2)
                / (2 * self.sigma**2)
            )

    """
    Misc
    
    """

    def _store(self):
        # Final simulation
        self.sim = sim_img_batch(
            self.cyts_opt,
            self.mems_opt,
            self.cytbg_opt,
            self.membg_opt,
            zerocap=self.zerocap,
            swish_factor=self.swish_factor,
        )

        # Images: remove padded regions and rescale
        self.straight_images = [
            img.T[mask == 1].T * norm
            for img, mask, norm in zip(self.target, self.masks, self.norms)
        ]
        self.straight_images_sim = [
            img.T[mask == 1].T * norm
            for img, mask, norm in zip(self.sim, self.masks, self.norms)
        ]

        # Save and rescale quantification results
        if self.zerocap:
            _m = self.mems_opt * sigmoid(self.swish_factor * self.mems_opt)
            _c = self.cyts_opt * sigmoid(self.swish_factor * self.cyts_opt)
            self.mems = [
                m[mask == 1] * norm for m, mask, norm in zip(_m, self.masks, self.norms)
            ]
            self.cyts = [
                c * norm * np.ones(int(np.sum(mask)))
                for c, mask, norm in zip(_c, self.masks, self.norms)
            ]
        else:
            self.mems = [
                m[mask == 1] * norm
                for m, mask, norm in zip(self.mems_opt, self.masks, self.norms)
            ]
            self.cyts = [
                c * norm * np.ones(int(np.sum(mask)))
                for c, mask, norm in zip(self.cyts_opt, self.masks, self.norms)
            ]

        # Reference profiles
        self.cytbg = self.cytbg_opt
        self.membg = self.membg_opt

    """
    Interactive
    
    """

    def plot_losses(self, log: bool = False):
        """
        Plot loss curves (one line for each image)

        Args:
            log: if True, plot the logarithm of losses
        """
        fig, ax = plt.subplots()
        losses = np.log10(self.losses.T) if log else self.losses.T
        ylabel = "log10(Mean square error)" if log else "Mean square error"
        ax.plot(losses)
        ax.set_xlabel("Descent step")
        ax.set_ylabel(ylabel)
        return fig, ax


def sim_img_batch(cyts, mems, cytbg, membg, zerocap, swish_factor):
    """
    [nimgs, thickness, nfits]

    """

    # Cytbg: expand dimensions
    _cytbg = jnp.expand_dims(jnp.expand_dims(cytbg, axis=0), axis=-1)

    # Cyts: expand dimensions
    _cyt = jnp.expand_dims(jnp.expand_dims(cyts, axis=-1), axis=-1) * jnp.expand_dims(
        jnp.expand_dims(jnp.ones(mems.shape[1], dtype=jnp.float32), axis=0), axis=0
    )

    # Cytoplasm zerocap
    if zerocap:
        _cyt *= sigmoid(swish_factor * _cyt)

    # Cytoplasmic signal contribution
    cyt_total = _cytbg * _cyt

    # Membg: expand dimensions
    _membg = jnp.expand_dims(jnp.expand_dims(membg, axis=0), axis=-1)

    # Membrane: expand dimensions
    _mem = jnp.expand_dims(mems, axis=1)

    # Membrane zerocap
    if zerocap:
        _mem = _mem * sigmoid(swish_factor * _mem)

    # Membrane signal contribution
    mem_total = _membg * _mem

    # Sum membrane and cytoplasmic contributions
    return jnp.add(cyt_total, mem_total)


def masked_loss_function(sim, target, masks):
    sq_errors = (sim - target) ** 2  # calculate errors
    mse = jnp.sum(sq_errors * jnp.expand_dims(masks, 1), axis=[1, 2]) / jnp.sum(
        masks, axis=1
    )  # masked average
    return mse


# Legacy alias
ImageQuant2 = ImageQuantFlexi
