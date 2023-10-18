# PAR Segmentation

[![CC BY 4.0][cc-by-shield]][cc-by]
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests](https://github.com/goehringlab/par-segmentation/actions/workflows/test.yaml/badge.svg)](https://github.com/goehringlab/par-segmentation/actions/workflows/test.yaml)
[![PyPi version](https://badgen.net/pypi/v/par-segmentation/)](https://pypi.org/project/par-segmentation)
[![Documentation Status](https://readthedocs.org/projects/par-segmentation/badge/?version=latest)](https://par-segmentation.readthedocs.io/en/latest/?badge=latest)
[![run with docker](https://img.shields.io/badge/run%20with-docker-0db7ed?logo=docker)](https://www.docker.com/)
[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?logo=anaconda)](https://docs.conda.io/en/latest/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/goehringlab/par-segmentation/HEAD?filepath=%2Fscripts/Tutorial.ipynb)
[![codecov](https://codecov.io/gh/goehringlab/par-segmentation/branch/master/graph/badge.svg?token=QCFC9AWK0R)](https://codecov.io/gh/goehringlab/par-segmentation)


Tools for segmenting, straightening and quantifying the cortex of cells.
Works by combining spline-based segmentation with a custom quantification model, using a gradient descent optimisation procedure.
Designed primarily for membrane-bound PAR proteins in C. elegans zygotes.

<p align="center">
    <img src="https://raw.githubusercontent.com/goehringlab/par-segmentation/master/scripts/Figs/animation.gif" width="100%" height="100%"/>
</p>

Advantages:

- Combine segmentation and membrane quantification in a single step
- No ground truth training data required
- Works for single snapshots or timelapse movies

Disadvantages:

- Requires a small amount of manual annotation for every image
- A little slow compared to some other segmentation methods 


## Instructions

As a first step, I would recommend checking out the [tutorial notebook](https://nbviewer.org/github/goehringlab/par-segmentation/blob/master/scripts/Tutorial.ipynb). To run the notebook interactively you have a few options:

#### Option 1: Binder

To run in the cloud using Binder, click here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/goehringlab/par-segmentation/HEAD?filepath=%2Fscripts/Tutorial.ipynb)

(Please note that it may take several minutes to open the notebook)

#### Option 2: Docker

Step 1: Make sure [Docker](https://www.docker.com/products/docker-desktop/) is installed and open on your machine 

Step 2: Download and run the Docker container: 

    docker run --rm -p 8888:8888 tsmbland/par-segmentation

Once the Docker image has finished downloading, this will print two URLs at the bottom for you to copy and paste into your web browser to open up Jupyter (please try both)

Step 3: Navigate to _scripts/Tutorial.ipynb_ to run the notebook

Step 4: When finished, delete the image:

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
    <img src="https://raw.githubusercontent.com/goehringlab/par-segmentation/master/docs/model schematic.png" width="100%" height="100%"/>
</p>

Cortex positions are modelled as a spline with a user-specified number of evenly spaced knots which are optimised by gradient descent:

<p align="center">
    <img src="https://raw.githubusercontent.com/goehringlab/par-segmentation/master/scripts/Figs/spline.png" width="80%" height="80%"/>
</p>

Cross-cortex intensity profiles at each position around the cortex are modelled as the sum of distinct cytoplasmic and membrane signal components:
an error function and Gaussian function respectively, representing the expected shape of a step and a point convolved by a Gaussian point spread function (PSF) in one dimension:

<p align="center">
    <img src="https://raw.githubusercontent.com/goehringlab/par-segmentation/master/scripts/Figs/profiles.png" width="100%" height="100%"/>
</p>

The program learns the amplitude of these two components at each position around the cortex, so can serve as a quantification tool as well as a segmentation tool:

<p align="center">
    <img src="https://raw.githubusercontent.com/goehringlab/par-segmentation/master/scripts/Figs/animation2.gif" width="100%" height="100%"/>
</p>

Modelling the PSF as a Gaussian, and ignoring out-of-focus contributions, is a clear simplification of reality, but is a close enough approximation for many purposes (e.g. if you're interested in relative concentrations rather than absolute concentrations). Nevertheless, one can relax these assumptions (with some added caveats) if higher accuracy is required. 
See [here](https://github.com/tsmbland/discco).


## Publications

This package has been used in the following publications: 

- [Optimized dimerization of the PAR-2 RING domain drives cooperative and selective membrane recruitment for robust feedback-driven cell polarization.](https://www.biorxiv.org/content/10.1101/2023.08.10.552581v1) BioRxiv (2023).

- [Design principles for selective polarization of PAR proteins by cortical flows.](https://rupress.org/jcb/article/222/8/e202209111/214138/Design-principles-for-selective-polarization-of) Journal of Cell Biology (2023).

- [Cleavage furrow-directed cortical flows bias mechanochemical pathways for PAR polarization in the C . elegans germ lineage.](https://www.biorxiv.org/content/10.1101/2022.12.22.521633v1.abstract) BioRxiv (2022).

- [An analog sensitive allele permits rapid and reversible chemical inhibition of PKC-3 activity in C. elegans.](https://www.micropublication.org/journals/biology/micropub-biology-000610) Micropublication Biology (2022).

- [SAIBR : A simple, platform-independent method for spectral autofluorescence correction.](https://journals.biologists.com/dev/article/149/14/dev200545/276004/SAIBR-a-simple-platform-independent-method-for) Development (2022).

_To add your paper to this list, please use the issues form, or create a pull request_


## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
