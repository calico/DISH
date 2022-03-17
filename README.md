<p>
    <a href="https://docs.calicolabs.com/python-template"><img alt="docs: Calico Docs" src="https://img.shields.io/badge/docs-Calico%20Docs-28A049.svg"></a>
    <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# DISH Scoring

![](https://github.com/calico/myproject)

## Overview

This tool applies ML models to the analysis of DEXA images for measuring
bone changes that are related to diffuse idiopathic skeletal hyperostosis (DISH).

## [DISH Analysis: Methods Description](docs/analysis.md)

## [Developer Documentation](docs/developer.md)

## Installation
The recommended build environment for the code is to have [Anaconda](https://docs.anaconda.com/anaconda/install/) installed and then to create a conda environment for python 3 as shown below:

```
conda create -n dish python=3.7
```

Once created, activate the environment and install all the needed libraries as follows: 

``` 
conda activate dish
pip install -r requirements.txt
```

## Usage 
An example for a recommended invokation of the code:

```
python scoreSpines.py -i <dir of imgs> -o <out file> --aug_flip --aug_one
```
### [Detailed Usage Instructions](docs/getstarted.md)


## License

See LICENSE

## Maintainers

See CODEOWNERS
