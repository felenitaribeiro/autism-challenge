# Algorithmic Bias Investigation of Winners of the IMaging-PsychiAtry Challenge: predicting autism

[![Build Status](https://travis-ci.org/ramp-kits/autism.svg?branch=master)](https://travis-ci.org/ramp-kits/autism)

## Getting started

This starting kit requires Python and the following dependencies:

* `numpy<1.21.6`
* `scipy`
* `pandas>=0.21`
* `scikit-learn>=0.19,<0.22`
* `matplolib`
* `seaborn`
* `nilearn<0.8`
* `jupyter`
* `ramp-workflow`
* `cloudpickle`
* `keras`
* `tensorflow=2.5`

Therefore, we advise you to install [Anaconda
distribution](https://www.anaconda.com/download/) which include almost all
dependencies.

Only `nilearn` and `ramp-workflow` are not included by default in the Anaconda
distribution. They will be installed from the execution of the notebook.

You can attempt to install these by executing the jupyter notebook, from the root directory using:

```
jupyter notebook autism_starting_kit.ipynb
```


## Recommended install using `conda` (optional)

We provide both an `environment.yml` file which can be used with `conda` to
create a clean environment and install the necessary dependencies.

```
conda env create -f environment.yml
```

Then, you can activate the environment using:

```
source activate autism
```

for Linux and MacOS. In Windows, use the following command instead:

```
activate autism
```

## Launch on Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ramp-kits/autism/best_submissions?filepath=best_submissions.ipynb)
