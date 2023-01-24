# PAR Segmentation

[![CC BY 4.0][cc-by-shield]][cc-by]
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
[![PyPi version](https://badgen.net/pypi/v/par-segmentation/)](https://pypi.org/project/par-segmentation)

Given an image of a cell and a rough preliminary ROI, it learns both the coordinates of the cell cortex and spatial measures of membrane and cytoplasmic concentrations.
Designed primarily for use on images of PAR proteins in C. elegans zygotes.

## Introduction

Address the problem here. We have images of cells and want to quantify membrane signal

Can draw a sketch: image -> rough idea of what membrane profile should look like

Requires a two-step process:
- Segmentation to find the cell edge (sketch)
- Fitting cross-sectional profiles to a model (sketch)

Here I show that these steps can be combined in an end-to-end fashion to perform segmentation and quantification simulataneously

Advantages:
- less user workload
- more accurate

## Methods

Schematic of model

<p align="center">
    <img src="docs/model schematic.png" width="100%" height="100%"/>
</p>

Based on a model found in Gross et al.

More figures/animations showing the model in action

## Installation

    pip install par-segmentation

## Instructions

Binder link

## Publications

List of publications that used this code: KB micropub, Rukshala's paper, Filopodia paper?
(Nelio's paper, KB P1 paper, my paper, Joana's paper)

(Some of these use an older variant of the code)

## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

