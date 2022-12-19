import time
from .quantification_differential_evolution_single import ImageQuantDifferentialEvolutionSingle
from typing import Union, Optional
import numpy as np


class ImageQuantDifferentialEvolutionMulti:
    def __init__(self,
                 img: Union[np.ndarray, list],
                 roi: Union[np.ndarray, list] = None,
                 sigma: float = 2.0,
                 periodic: bool = True,
                 thickness: int = 50,
                 freedom: bool = 0.5,
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
