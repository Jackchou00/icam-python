# Python Implementation of the Image Color Appearance Model (iCAM)

English | [简体中文](ReadMe_zh.md)

## Introduction

Color appearance models, as an extension of colorimetry, have significant applications in image processing, particularly in tone mapping for high dynamic range (HDR) imaging.

The iCAM framework, proposed by Fairchild and colleagues, adapts color appearance models to image processing. The iCAM06 version incorporates state-of-the-art colorimetric methods and models from its time, such as CAT02, CAM02, and IPT. This framework has since been refined and extended, with derivatives like TMOz.

This project provides a Python implementation of the iCAM framework, based on the original MATLAB code for iCAM06.

Detailed research and model details can be found at: [iCAM06 Survey](https://jackchou00.com/posts/icam06-survey/)


## References

[1] J. Kuang, G. M. Johnson, and M. D. Fairchild, “iCAM06: A refined image appearance model for HDR image rendering,” Journal of Visual Communication and Image Representation, vol. 18, no. 5, pp. 406–414, Oct. 2007, doi: 10.1016/j.jvcir.2007.06.003.

[2] I. Mehmood, M. Zhou, M. U. Khan, and M. Luo, “CIECAM16-based Tone Mapping of High Dynamic Range Images,” Color and Imaging Conference, vol. 31, pp. 102–107, Nov. 2023, doi: 10.2352/CIC.2023.31.1.20.

## Input and Output

iCAM accepts an XYZ image as input, XYZ here refers to the original absolute XYZ tristimulus, without chromatic adaptation or normalization. It can be gainned from camera and coresponding CCM (Color Correction Matrix) or from a image by inverse pipeline.

The output of iCAM is a image with the same size as the input. But XYZ values have been compressed and can be converted to sRGB or other color spaces, then displayed on a monitor.

## Issues

Some issues have been found in the original iCAM06 code.

- The modified CAT02 with a D multiplied by 0.3. Incorrect handling of the relationship between the adaptation field and tristimulus values in a diagonal matrix (due to incorrect normalization) can lead to changes in the brightness of the image after color adaptation.
- The calculation of D in CAT02 is incorrect. Should be $-(L_A+42)$ instead of $-(L_A-42)$.

## Files

The project is currently undergoing structural improvements to enhance modularization and readability, aiming for a more organized and maintainable codebase.

I have divided the repository into the following sections, following the structure of iCAM:

## Repository Structure

```text
icam-python/
├── icam06/                 # Primary integrated package
│   ├── cat.py              # Chromatic adaptation models
│   ├── rgb_xyz.py          # Universal RGB/XYZ conversion
│   ├── tc.py               # Tone compression
│   └── ...
├── chromatic_adaptation/     # Legacy modules (backward compatibility)
├── colour_space_conversion/  # Legacy modules
├── research/               # Research notes and survey
├── example/                # Data files and output
├── main.py                 # Main demonstration script
└── pyproject.toml          # uv configuration
```

