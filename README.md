# DiSCo-Seg: Differentiable simulation for cortex segmentation

An unsupervised gradient-descent based method for cell cortex segmentation, based on differentiable simmulation.
Given an image of a cell and a rough preliminary ROI, disco-seg provides both the coordinates of the cell cortex and spatial measures of membrane and cytoplasmic concentrations.
Designed primarily for use on images of C. elegans embryos.

## Introduction

Address the problem here. We have images of cells and want to quantify membrane signal

Can draw a sketch: image -> rough idea of what membrane profile should look like

Requires a two step process:
- Segmentation to find the cell edge (sketch)
- Fitting cross sectional profiles to a model (sketch)

Here I show that these steps can be combined in an end-to-end fashion to perform segmentation and quantification simulataneously

Advantages:
- less user workload
- more accurate

## Differentiable model

Schematic of model

<p align="center">
    <img src="docs/model schematic.png" width="100%" height="100%"/>
</p>

Based on a model found in Gross et al

More figures/animations showing the model in action


## Applications

Some examples from the papers it's been used in

Could try it on some images of cells from other systems e.g. yeast?

## Installation

Instructions on how to create conda environment


## Instructions

Link to a tutorial notebook. Also make a binder link

## Publications

List of publications that used this code: KB micropub, Rukshala's paper, Filopodia paper?
(Nelio's paper, KB P1 paper, my paper, Joana's paper)

(Some of these use an older variant of the code)

