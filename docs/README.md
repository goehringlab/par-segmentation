# DISCCo: Differentiable Image Simulation of the Cell Cortex

[![CC BY 4.0][cc-by-shield]][cc-by]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyPi version](https://badgen.net/pypi/v/discco/)](https://pypi.org/project/discco)

Quantification of membrane and cytoplasmic concentrations based on differentiable simulation of cell cortex images.
Designed for use on images of PAR proteins in C. elegans zygotes.

This extends on the segmentation and straightening algorithm described [here](https://github.com/tsmbland/par-segmentation), and uses straightened cortices obtained by that method as input.

## Installation

    pip install discco

## Methods

Our method is adapted from previous methods that model intensity profiles perpendicular to the membrane as the sum of distinct cytoplasmic and membrane signal components (Gross et al., 2018; Reich et al., 2019). Typically these two components are modelled as an error-function and Gaussian function respectively, representing the expected shape of a step and a point convolved by a Gaussian point spread function (PSF) in one dimension. Using this model, one can generate simulated images of straightened cortices as the sum of two tensor products which represent distinct membrane and cytoplasmic signal contributions (Figure 1):

sim = c<sub>cyt</sub> ⊗ s<sub>cyt</sub> + c<sub>mem</sub> ⊗ s<sub>mem</sub>

where c<sub>cyt</sub> and c<sub>mem</sub> are cytoplasmic and membrane concentration profiles and s<sub>cyt</sub> and s<sub>mem</sub> are, by default, error-function and Gaussian profiles. We impose the constraint that the cytoplasmic concentration c<sub>cyt</sub> is uniform throughout each image.
<p align="center">
    <img src="https://raw.githubusercontent.com/tsmbland/discco/master/docs/schematic.png" width="100%" height="100%"/>
    <i>Figure 1: Schematic of differentiable model for image quantification</i>
</p>
<br>

Using a differentiable programming paradigm, the input parameters to the model can be iteratively adjusted by backpropagation to minimize the mean squared error between simulated images and ground truth images.
As well as allowing the image-specific concentration parameters (c<sub>cyt</sub> and c<sub>mem</sub>) to be learnt, this procedure also allows the global signal profiles s<sub>cyt</sub> and s<sub>mem</sub> to be optimised and take any arbitrary form, allowing the model to generalise beyond a simple Gaussian PSF model and account for complex sample-specific light-scattering behaviors. In practice we find that this additional flexibility is necessary to minimise model bias and prevent underfitting:

<p align="center">
    <img src="https://raw.githubusercontent.com/tsmbland/discco/master/docs/simulation comparison.png" width="80%" height="80%"/>
    <br>
    <i>Figure 2: Example of ground truth and simulated images. Naive model refers to a mechanistic optical model with a Guassian PSF. Gaussian noise has been added to simulated images to allow for closer visual comparison to the ground truth image.</i>
   
</p>  
<br>

An additional step, described in the paper, puts the cytoplasmic and membrane concentrations outputted by the model into biologically meaningful units, which has great utility for mathematical models.

For full details of the model and training procedures, see the paper:

[Optimized dimerization of the PAR-2 RING domain drives cooperative and selective membrane recruitment for robust feedback-driven cell polarization](https://www.biorxiv.org/content/10.1101/2023.08.10.552581v1) (preprint)

And the accompanying [GitHub repository](https://github.com/goehringlab/2023-Bland-par2).

Limitations:
- Relies on a few assumptions about the system (uniform cytoplasmic concentration, rotational symmetry)
- Requires several calibrations with a few different samples


## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

