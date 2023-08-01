# PAR Segmentation

[![CC BY 4.0][cc-by-shield]][cc-by]
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
[![PyPi version](https://badgen.net/pypi/v/par-segmentation/)](https://pypi.org/project/par-segmentation)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tsmbland/par-segmentation/HEAD?filepath=%2Fscripts/Tutorial.ipynb)
[![run with docker](https://img.shields.io/badge/run%20with-docker-0db7ed?logo=docker)](https://www.docker.com/)
[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?logo=anaconda)](https://docs.conda.io/en/latest/)

Tools for segmenting, straightening and quantifying the cortex of cells.
Works by combining spline-based segmentation with a custom quantification model, using a gradient descent optimisation procedure.
Designed primarily for membrane-bound PAR proteins in C. elegans zygotes.

<p align="center">
    <img src="https://raw.githubusercontent.com/tsmbland/par-segmentation/master/scripts/Figs/animation.gif" width="100%" height="100%"/>
</p>


## Instructions

As a first step, I would recommend checking out the [tutorial notebook](https://nbviewer.org/github/tsmbland/par-segmentation/blob/master/scripts/Tutorial.ipynb). To run the notebook interactively you have a few options:

#### Option 1: Binder

To run in the cloud using Binder, click here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tsmbland/par-segmentation/HEAD?filepath=%2Fscripts/Tutorial.ipynb)

(Please note that it may take several minutes to open the notebook)

#### Option 2: Docker

Step 1: With [Docker](https://www.docker.com/products/docker-desktop/) open on your machine,  pull the image (copy and paste into the terminal)

    docker pull tsmbland/par-segmentation

Step 2: Run the docker container (copy and paste into the terminal)

    docker run -p 8888:8888 tsmbland/par-segmentation

This will print a URL at the bottom for you to copy and paste into your web browser to open up Jupyter

Step 3: When finished, delete the container and image
    
    docker container prune -f
    docker image rm tsmbland/par-segmentation

#### Option 3: Conda

You can use the environment.yml file to set up a [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment on your machine from which the notebook can be run

    conda env create -f environment.yml
    conda activate par-segmentation
    jupyter notebook


## Installation

To explore further and incorporate into your own analysis pipelines, you can install the package using pip:

    pip install par-segmentation

## Methods

Starting with an initial rough manual ROI of the cell edge, the cortex of the image is straightened (Step 1).
The program then attempts to mimic this straightened image by differentiable simulation (Step 2).
In doing so, it learns the position of the cortex, which enables the ROI to be adjusted (Step 3) and the cortex re-straightened.

<p align="center">
    <img src="https://raw.githubusercontent.com/tsmbland/par-segmentation/master/docs/model schematic.png" width="100%" height="100%"/>
</p>

Cortex positions are modelled as a spline with a user-specified number of evenly spaced knots which are optimised by gradient descent:

<p align="center">
    <img src="https://raw.githubusercontent.com/tsmbland/par-segmentation/master/scripts/Figs/spline.png" width="80%" height="80%"/>
</p>

Cross-cortex intensity profiles at each position around the cortex are modelled as the sum of distinct cytoplasmic and membrane signal components:
an error function and Gaussian function respectively, representing the expected shape of a step and a point convolved by a Gaussian point spread function (PSF) in one dimension:

<p align="center">
    <img src="https://raw.githubusercontent.com/tsmbland/par-segmentation/master/scripts/Figs/profiles.png" width="100%" height="100%"/>
</p>

The program learns the amplitude of these two components at each position around the cortex, so can serve as a quantification tool as well as a segmentation tool:

<p align="center">
    <img src="https://raw.githubusercontent.com/tsmbland/par-segmentation/master/scripts/Figs/animation2.gif" width="100%" height="100%"/>
</p>

The model is a slight simplification of reality, and doesn't account for the possibility of a non-Gaussian PSF and complex 3D light-scattering behaviours, but is a close enough approximation for many purposes. 
Nevertheless, one can relax these assumptions if higher quantification accuracy is required. 
See [here](https://github.com/tsmbland/discco) for an extension of the method designed for more accurate quantification.


## Publications

Code in this repository has been used in the following publications for PAR protein segmentation and/or quantification: 

Illukkumbura, R., Hirani, N., Borrego-Pinto, J., Bland, T., Ng, K., Hubatsch, L., McQuade, J., Endres, R.G., and Goehring, N.W. (2022). Design principles for selective polarization of PAR proteins by cortical flows. BioRxiv 2022.09.05.506621.

Ng, K., Hirani, N., Bland, T., Borrego-pinto, J., and Goehring, N.W. (2022a). Cleavage furrow-directed cortical flows bias mechanochemical pathways for PAR polarization in the C . elegans germ lineage. BioRxiv 1–32.

Ng, K., Bland, T., Hirani, N., and Goehring, N.W. (2022b). An analog sensitive allele permits rapid and reversible chemical inhibition of PKC-3 activity in C . elegans. MicroPublication Biol.

Rodrigues, N.T.L., Bland, T., Borrego-Pinto, J., Ng, K., Hirani, N., Gu, Y., Foo, S., and Goehring, N.W. (2022). SAIBR : A simple, platform-independent method for spectral autofluorescence correction. Development.




## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

