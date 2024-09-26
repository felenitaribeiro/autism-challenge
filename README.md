# Algorithmic Bias Investigation of Winners of the IMaging-PsychiAtry Challenge: predicting autism

Note this is a fork from the original challenge and has been utilised as part of a thesis submission. The contents of this fork have been heavily editted during the investigation. For original documents pease go upstream into the "main" brainch or further to https://github.com/neuroanatomy/autism-challenge
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


## Recommended install using `conda` (trust me, use this method)

I have found an environment able to run all submissions. This `environment.yml` file which can be used with `conda` to create a clean environment and install the necessary dependencies.

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