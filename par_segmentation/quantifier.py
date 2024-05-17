import numpy as np


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
        method: 'GD' for gradient descent or 'DE' for differential evolution. The former is highly recommended, the latter works but is much slower and no longer maintained
    """

    def __init__(
        self,
        img: np.ndarray | list,
        roi: np.ndarray | list,
        method: str = "GD",
        **kwargs,
    ):
        # Set up quantifier
        self.method = method
        if self.method == "GD":
            from .model_gd import ImageQuantGradientDescent

            self.iq = ImageQuantGradientDescent(
                img=img,
                roi=roi,
                **kwargs,
            )

        elif self.method == "DE":
            from .model_de import ImageQuantDifferentialEvolutionMulti

            self.iq = ImageQuantDifferentialEvolutionMulti(
                img=img,
                roi=roi,
                **kwargs,
            )

        elif self.method == "flexi":
            from .model_flexi import ImageQuantFlexi

            self.iq = ImageQuantFlexi(
                img=img,
                roi=roi,
                **kwargs,
            )

        else:
            raise Exception('method argument must be "GD", "DE" or "flexi"')

    def __getattr__(self, name):
        return getattr(self.iq, name)
