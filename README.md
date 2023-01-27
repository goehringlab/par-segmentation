# PAR Segmentation

[![CC BY 4.0][cc-by-shield]][cc-by]
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
[![PyPi version](https://badgen.net/pypi/v/par-segmentation/)](https://pypi.org/project/par-segmentation)

Tools for segmenting and straightening the cortex of cells from midplane images using a gradient descent algorithm.
Designed primarily for use on images of PAR proteins in C. elegans zygotes.

<p align="center">
    <img src="https://raw.githubusercontent.com/tsmbland/par-segmentation/master/scripts/animation.gif" width="100%" height="100%"/>
</p>

## Methods

Starting with an initial rough manual ROI of the cell edge, the cortex of the image is straightened (1).
The program then attempts to mimic this straightened image by differentiable simulation (2).
In doing so, it learns the position of the cortex, which enables the ROI to be iteratively adjusted (3) and the cortex re-straightened:

<p align="center">
    <img src="https://raw.githubusercontent.com/tsmbland/par-segmentation/master/docs/model schematic.png" width="100%" height="100%"/>
</p>

The program additionally outputs parameters related to cytoplasmic and membrane concentrations, so can serve as a quantification tool as well as a segmentation tool.
See also [here](https://github.com/tsmbland/discco) for an extension of the method designed for more accurate quantification of cytoplasmic and membrane concentrations.

## Installation

    pip install par-segmentation

## Instructions

Binder link (TO DO)

## Publications

Utilised in the following publications: 

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

