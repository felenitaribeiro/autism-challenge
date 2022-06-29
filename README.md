# Information and code for the IMaging-PsychiAtry Challenge setup and analyses

### Accompanying repo for our paper _Insights from an autism imaging biomarker challenge: promises and threats to biomarker discovery_
available on [medRxiv](https://www.medrxiv.org/content/10.1101/2021.11.24.21266768v3).

Nicolas Traut, Katja Heuer, Guillaume Lemaître, Anita Beggiato, David Germanaud, Monique Elmaleh, Alban Bethegnies, Laurent Bonnasse-Gahot, Weidong Cai, Stanislas Chambon, Freddy Cliquet, Ayoub Ghriss, Nicolas Guigui, Amicie de Pierrefeu, Meng Wang, Valentina Zantedeschi, Alexandre Boucaud, Joris van den Bossche, Balázs Kegl, Richard Delorme, Thomas Bourgeron, Roberto Toro, & Gaël Varoquaux.
<br />
<br />
<br />
![](https://drive.google.com/uc?id=15UgK8NMCX2ZmLH-hYE4XrCk5J7NpP2Ry)
<br />
<br />

# Website describing the vision of the challenge to potential participants

All necessary information about the data, the scientific implications and participation to the challenge have been provided on [the IMPAC challenge website](https://paris-saclay-cds.github.io/autism_challenge).


# Preparation of the challenge

## Starting kit for the IMaging-PsychiAtry Challenge: predicting autism

[![Build Status](https://travis-ci.org/ramp-kits/autism.svg?branch=master)](https://travis-ci.org/ramp-kits/autism)

## Getting started

This starting kit requires Python and the following dependencies:

* `numpy<1.20`
* `scipy`
* `pandas>=0.21`
* `scikit-learn>=0.19,<=0.21`
* `nilearn<0.8`
* `matplolib`
* `seaborn`
* `jupyter`
* `ramp-workflow==0.2.1`

Therefore, we advise you to install [Anaconda
distribution](https://www.anaconda.com/download/) which include almost all
dependencies.

Only `nilearn` and `ramp-workflow` are not included by default in the Anaconda
distribution. They will be installed from the execution of the notebook.

Execute the jupyter notebook, from the root directory using:

```
jupyter notebook autism_starting_kit.ipynb
```


## Advanced install using `conda` (optional)

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

# The 10 best submissions

Code for the 10 best submissions is available in this repo in the branch `best_submissions`. It holds the feature extractor and classifier scripts from the final submissions that scored best.


# Post-hoc analyses

All scripts used for the data analyses and figures presented in our paper can be found in the [autism-challenge-analyses repo](https://github.com/neuroanatomy/autism-challenge-analyses).


