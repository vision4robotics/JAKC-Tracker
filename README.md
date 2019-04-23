# JAKC-tracker

| **Test passed**                                              |
| ------------------------------------------------------------ |
| [![matlab-2017b](https://img.shields.io/badge/matlab-2017b-yellow.svg)](https://www.mathworks.com/products/matlab.html) [![MatConvNet-1.0--beta25](https://img.shields.io/badge/MatConvNet-1.0--beta25%20-blue.svg)](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz) ![CUDA-8.0](https://img.shields.io/badge/CUDA-8.0-green.svg) |

> Matlab implementation of *Joint Learning-Aware Kernelized Correlator for UAV Tracking via Dynamic Context Constraint with Convolutional Features* (JAKC-tracker).

## Instructions

1. Download `imagenet-vgg-verydeep-19.mat` from [here](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) and put it in `/model`.
2. Download matconvnet toolbox [here](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz) and put it in `/external`.
3. Run `JAKC_Demo.m`

Note: the original version is using CPU to run the whole program. If GPU version is required, just change `false` in the following lines in `JAKC_Demo.m` to `true`:

```
global enableGPU;
enableGPU = false;

vl_setupnn();
vl_compilenn('enableGpu', false);
```

## Acknowledgements

Partly borrowed from [KCC](https://github.com/wang-chen/KCC/tree/master/tracking).

