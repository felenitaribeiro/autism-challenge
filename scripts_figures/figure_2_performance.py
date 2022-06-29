import os
from itertools import product

import numpy as np
import pandas as pd
import matplotlib

import sys
sys.path.append("..")

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

from utils import load_train_test_prediction
from utils import load_train_test_prediction_aims
from utils import load_train_test_blended_prediction
from utils import load_train_test_blended_prediction_lc
from utils import compute_roc_auc_score
from problem import get_test_data


with sns.plotting_context("poster"):

    # Probability of ASD for patients and controls
    _, _, y_true_test, y_pred_test = load_train_test_blended_prediction("original")
    cat = pd.Series(["Controls", "Patients"])
    pred_df = pd.DataFrame({"Group":cat[y_true_test], "Prediction":y_pred_test[:,1]})
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="Prediction", y="Group", data=pred_df, linewidth=0)
    sns.swarmplot(x="Prediction", y="Group", data=pred_df, color='black', alpha=.6)
    plt.savefig('../figures/fig_2a_patients_vs_controls.svg', bbox_inches="tight")


    # ROC curve combined all data
    team_name = [
        "abethe",
        "amicie",
        "ayoub.ghriss",
        "mk",
        "nguigui",
        "pearrr",
        "Slasnista",
        "vzantedeschi",
        "wwwwmmmm",
    ]
    group = "original"
    all_submissions = [
        tn + "_" + mt for tn, mt in product(team_name, [group])
    ]

    y_true_train, y_pred_train, y_true_test, y_pred_test = zip(
        *[load_train_test_prediction(sub) for sub in all_submissions]
    )
    df = pd.DataFrame(
        {
            "y_true_train": y_true_train,
            "y_pred_train": y_pred_train,
            "y_true_test": y_true_test,
            "y_pred_test": y_pred_test,
        },
        index=all_submissions,
    )
    df.index = df.index.str.split("_", n=1, expand=True)
    df = df.reset_index()
    df = df.rename(columns={"level_0": "team", "level_1": "modality"})
    df["modality"] = df["modality"].str.replace("_", " + ")
    df = df[df["modality"] == "original"]

    plt.figure(figsize=(10, 10))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for _, r in df.iterrows():
        fpr, tpr, _ = roc_curve(
            r["y_true_test"], r["y_pred_test"][:, 1]
        )
        # interpolate the tpr on a linear fpr space
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plot the roc for each submission
        plt.plot(fpr, tpr, ":", alpha=0.5, lw=1.5, color="tab:blue")

    # plot the chance line
    plt.plot(
        [0, 1], [0, 1], color="black", linestyle="--", lw=2, alpha=0.8,
        label="Chance"
    )

    # plot the ROC curve of the blended submission
    _, _, y_true_test_blended, y_pred_test_blended = load_train_test_blended_prediction(group)
    blended_fpr, blended_tpr, _ = roc_curve(y_true_test_blended, y_pred_test_blended[:, 1])
    blended_auc = auc(blended_fpr, blended_tpr)
    plt.plot(blended_fpr, blended_tpr, color='tab:orange',
             label=r'Blended ROC (AUC = %0.3f)' % (blended_auc),
             alpha=.8)

    # plot the screening control point
    screening_tpr = 0.85
    confirmatory_fpr = 0.03
    screening_idx = np.where(blended_tpr >= screening_tpr)[0][0]
    confirmatory_idx = np.where(blended_fpr <= confirmatory_fpr)[0][-1]
    plt.scatter(
        blended_fpr[[confirmatory_idx, screening_idx]],
        blended_tpr[[confirmatory_idx, screening_idx]],
        color=['tab:red', 'tab:green'],
        s=100,
        alpha=.8,
        zorder=3
    )
    plt.vlines(
            blended_fpr[[confirmatory_idx, screening_idx]],
            ymin=0, ymax=blended_tpr[[confirmatory_idx, screening_idx]],
            color=['tab:red', 'tab:green'],
            linestyle='--', 
            lw=2,
            alpha=.8,
            zorder=3
    )
    plt.hlines(
            blended_tpr[[confirmatory_idx, screening_idx]],
            xmin=0,
            xmax=blended_fpr[[confirmatory_idx, screening_idx]],
            color=['tab:red', 'tab:green'],
            linestyle='--',
            lw=2,
            alpha=.8,
            zorder=3
    )

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate", size=20)
    plt.ylabel("True Positive Rate")
    plt.title(
        "Prediction accuracy", size=20, fontweight="bold", y=1.12
    )
    plt.legend(
        loc=(0.53, 0.04),
        frameon=False,
        handlelength=1,
        handletextpad=0.5,
        prop={"size": 14},
    )
    plt.savefig('../figures/fig_2b_roc_{}.svg'.format(group), bbox_inches="tight")

    # Overall results modalities
    team_name = [
        "abethe",
        "amicie",
        "ayoub.ghriss",
        "mk",
        "nguigui",
        "pearrr",
        "Slasnista",
        "vzantedeschi",
        "wwwwmmmm",
    ]
    modality_type = [
        "anatomy",
        "functional",
        "anatomy_functional",
        "anatomy_functional_age_sex",
    ]
    all_submissions = [
        tn + "_" + mt for tn, mt in product(team_name, modality_type)
    ]

    # compute the ROC-AUC for training and testing and create a dataframe
    roc_auc_train, roc_auc_test = zip(
        *[
            compute_roc_auc_score(*load_train_test_prediction(sub))
            for sub in all_submissions
        ]
    )
    df_roc_auc = pd.DataFrame(
        {"ROC AUC train": roc_auc_train, "ROC AUC test": roc_auc_test},
        index=all_submissions,
    )
    # create a separate column for the team and the modality
    df_roc_auc.index = df_roc_auc.index.str.split("_", n=1, expand=True)
    df_roc_auc = df_roc_auc.reset_index()
    df_roc_auc = df_roc_auc.rename(
        columns={"level_0": "team", "level_1": "modality"}
    )

    # make a plot only with the testing ROC AUC groupy by modality
    # clean the name given to the modality
    df_roc_auc["modality"] = df_roc_auc["modality"].str.replace("_", " + ")
    df_roc_auc["modality"] = df_roc_auc["modality"].replace(
        "anatomy", "anatomy (cortical thickness)"
    )
    df_roc_auc["modality"] = df_roc_auc["modality"].replace(
        "functional", "functional (resting-state fMRI)"
    )

    # Seaborn creates too much padding here
    _, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df_roc_auc, y="modality", x="ROC AUC test", whis=10.0
    )
    for i in range(2):
        ax.axhspan(2 * i + 0.5, 2 * i + 1.5, color=".9", zorder=0)
    ax.set_title(
        "Importance of different data modalities", size=20,
        fontweight="bold"
    )
    plt.title('Prediction score')
    plt.xlabel('ROC-AUC')
    plt.ylabel('')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.savefig('../figures/fig_2c_overall_comparison.svg', bbox_inches="tight")

    # Comparison of performance between datasets
    team_name = [
        "abethe",
        "amicie",
        "ayoub.ghriss",
        "mk",
        "nguigui",
        "pearrr",
        "Slasnista",
        "vzantedeschi",
        "wwwwmmmm",
    ]
    modality_type = ["original"]
    all_submissions = [
        tn + "_" + mt for tn, mt in product(team_name, modality_type)
    ]

    # Performance on private dataset
    y_true_train, y_pred_train, y_true_test, y_pred_test = zip(
        *[load_train_test_prediction(sub) for sub in all_submissions]
    )
    df = pd.DataFrame(
        {
            "y_true_train": y_true_train,
            "y_pred_train": y_pred_train,
            "y_true_test": y_true_test,
            "y_pred_test": y_pred_test,
        },
        index=all_submissions,
    )
    df.index = df.index.str.split("_", n=1, expand=True)
    df = df.reset_index()
    df = df.rename(columns={"level_0": "team", "level_1": "modality"})
    df["modality"] = df["modality"].str.replace("_", " + ")

    # Performance on EU-Aims dataset
    y_true_train, y_pred_train, y_true_test, y_pred_test = zip(*[load_train_test_prediction_aims(sub)
                                                                 for sub in all_submissions])

    df_aims = pd.DataFrame({'y_true_train': y_true_train, 'y_pred_train': y_pred_train,
                       'y_true_test': y_true_test, 'y_pred_test': y_pred_test},
                      index=all_submissions)

    df_aims.index = df_aims.index.str.split('_', n=1, expand=True)
    df_aims = df_aims.reset_index()
    df_aims = df_aims.rename(columns={"level_0": "team", "level_1": "modality"})
    df_aims["modality"] = df_aims["modality"].str.replace("_", " + ")


    df_combin = df.merge(df_aims, on=['team', 'modality'], suffixes=('_ramp', '_aims'))
    df_combin.head()
    
    # Compute the performance only for RDB vs the others
    # Find the index corresponding to RDB
    # rdb_idx = np.load("rdb_idx.npy")
    X_test, y_test = get_test_data("..")
    X_rdb_idx = X_test['participants_site']==24
    rdb_idx = X_test.index.values[X_rdb_idx]

    auc_ramp = []
    auc_aims = []
    auc_rdb = []
    auc_other = []
    for idx, serie in df_combin.iterrows():
        y_true_ramp = serie['y_true_test_ramp']
        y_pred_ramp = serie['y_pred_test_ramp']
        roc_auc_ramp = roc_auc_score(y_true_ramp, y_pred_ramp[:, 1])
        auc_ramp.append(roc_auc_ramp)

        y_true_aims = serie['y_true_test_aims']
        y_pred_aims = serie['y_pred_test_aims']
        roc_auc_aims = roc_auc_score(y_true_aims, y_pred_aims[:, 1])
        auc_aims.append(roc_auc_aims)
        
        y_true_rdb = serie['y_true_test_ramp'][X_rdb_idx]
        y_pred_rdb = serie['y_pred_test_ramp'][X_rdb_idx]
        roc_auc_rdb = roc_auc_score(y_true_rdb, y_pred_rdb[:, 1])
        auc_rdb.append(roc_auc_rdb)

        y_true_other = serie['y_true_test_ramp'][~X_rdb_idx]
        y_pred_other = serie['y_pred_test_ramp'][~X_rdb_idx]
        roc_auc_other = roc_auc_score(y_true_other, y_pred_other[:, 1])
        auc_other.append(roc_auc_other)

    df_combin['roc_auc_ramp'] = auc_ramp
    df_combin['roc_auc_aims'] = auc_aims
    df_combin['roc_auc_rdb'] = auc_rdb
    df_combin['roc_auc_other'] = auc_other

    # Add new column to the dataframe with the AUC
    auc_dict = {'roc_auc_ramp': 'ABIDE+RDB', 'roc_auc_other': 'ABIDE', 'roc_auc_rdb': 'RDB', 'roc_auc_aims': 'EU-Aims'}
    df_auc = df_combin[['team', 'modality', *auc_dict]]
    df_auc = df_auc.rename(columns=auc_dict)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=pd.melt(
            df_auc,
            id_vars=["team", "modality"],
            value_vars=[*auc_dict.values()],
        ),
        y="variable",
        x="value",
        hue='modality',
        whis=10.0,
    )
    for i in range(2):
        plt.axhspan(2 * i + 0.5, 2 * i + 1.5, color=".9", zorder=0)

    plt.title('Heterogeneity across sites', size=20, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('../figures/fig_2d_comparison_datasets_bis.svg', bbox_inches="tight")


    # Learning curve
    participants = [
        "abethe",
        "amicie",
        "ayoub.ghriss",
        "mk",
        "nguigui",
        "pearrr",
        "Slasnista",
        "vzantedeschi",
        "wwwwmmmm",
    ]

    data_dict = {'# subjects': [], 'iteration': [], 'ROC-AUC train': [], 'ROC-AUC test': []}
    for nsub in range(500, 1501, 250):
        for it in range(1,101):
            data = load_train_test_blended_prediction_lc(nsub, it)
            if data[0] is None:
                continue
            data_dict['# subjects'].append(nsub)
            data_dict['iteration'].append(it)
            auc_train, auc_test = compute_roc_auc_score(*data)
            data_dict['ROC-AUC train'].append(auc_train)
            data_dict['ROC-AUC test'].append(auc_test)

    df = pd.DataFrame(data_dict)
    df = df.set_index('# subjects')

    # define the function which will be used to fit the learning curve
    def fit_func(x, a, b):
        return 0.5 + a * (1 - np.exp(-b * np.sqrt(x)))

    # gradient of the fit function used for the delta method (can also be estimated with bootstrap)
    def fit_func_grad(x, a, b):
        return np.array([1 - np.exp(-b * np.sqrt(x)), a * np.sqrt(x) * np.exp(-b * np.sqrt(x))])

    # Fit an extrapolation function
    mean_lr = df.groupby('# subjects')['ROC-AUC test'].mean()
    std_lr = df.groupby('# subjects')['ROC-AUC test'].std()
    size_lr = df.groupby('# subjects')['ROC-AUC test'].size()
    sigma_lr = std_lr / np.sqrt(size_lr)
    max_lr_idx = mean_lr.index.max()
    popt, pcov = curve_fit(fit_func, mean_lr.index.values, mean_lr.values, p0=[1, 1/max_lr_idx],
                           sigma=sigma_lr, absolute_sigma=True, bounds=(0, [1, np.inf]))

    plt.figure(figsize=(10, 7))
    plt.plot(mean_lr.index.values, mean_lr.values, 'o-')
    plt.fill_between(std_lr.index,
                     mean_lr - std_lr,
                     mean_lr + std_lr,
                     alpha=0.2,
                     label=r'$\pm$ 1 std. dev.')
    x_range = np.arange(500, 2701, 1)
    fit_values = fit_func(x_range, *popt)
    fit_grad = fit_func_grad(x_range, *popt)
    fit_se = np.sqrt(np.diag(fit_grad.T @ pcov @ fit_grad))
    plt.plot(x_range, fit_values, 'r--',
         label=r'fit: $0.5 + a \times(1 - e^{-b \cdot \sqrt{ n }})$' + '\n with a={:.2f}, b={:.3f}'.format(*popt))
    plt.fill_between(x_range,
                     fit_values - fit_se,
                     fit_values + fit_se,
                     color='red',
                     alpha=0.15,
                     label=r'$\pm$ 1 std. err.')
    plt.hlines(0.5 + popt[0], xmin=500, xmax=2700, color='red', linewidth=2, linestyle='--', label='asymptotic AUC')
    sns.despine(offset=10)
    plt.ylabel('ROC-AUC')
    plt.xlabel('# subjects in training set')
    plt.ylim([0.72, 0.86])
    plt.xlim([500, 2700])
    plt.xticks(np.arange(500, 2001, 500))
    plt.legend(loc='lower right')
    plt.title('Learning curve for different sample sizes')
    plt.savefig('../figures/fig_2e_learning_curve.svg', bbox_inches="tight")
