import os
# from itertools import product

import numpy as np
import pandas as pd
import matplotlib

import sys
sys.path.append("..")

matplotlib.use("agg")
import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
# 
# from scipy.optimize import curve_fit
# from scipy import stats
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.metrics import roc_auc_score
# 
# from utils import load_train_test_prediction
# from utils import load_train_test_prediction_aims
# from utils import load_train_test_blended_prediction
# from utils import load_train_test_blended_prediction_lc
from utils import compute_roc_auc_score
from problem import get_train_data, get_test_data

def load_train_test_prediction_all(submission_name):
    path_store_pred = os.path.join('..', 'submissions_all', submission_name, 'training_output')
    
    if not os.path.exists(os.path.join(path_store_pred, 'y_pred__bagged_valid.csv')):
        return (None, None, None, None)
    
    y_pred_train = np.loadtxt(os.path.join(path_store_pred, 'y_pred__bagged_valid.csv'))
    y_pred_test = np.loadtxt(os.path.join(path_store_pred, 'y_pred__bagged_test.csv'))
    
    _, y_true_train = get_train_data('..')
    _, y_true_test = get_test_data('..')
    
    # remove y_train values for which we have no prediction because they never went in the test part of the cv
    y_true_train = y_true_train[~np.isnan(y_pred_train[:,1])]
    y_pred_train = y_pred_train[~np.isnan(y_pred_train[:,1]), :]
    
    return (y_true_train, y_pred_train, y_true_test, y_pred_test)

# with sns.plotting_context("poster"):
submissions_all = os.listdir('../submissions_all')
best_submissions = ['abethe', 'amicie', 'ayoub.ghriss', 'lbg', 'mk', 'nguigui', 'pearrr', 'Slasnista', 'vzantedeschi', 'wwwwmmmm']

y_true_train, y_pred_train, y_true_test, y_pred_test = zip(*[load_train_test_prediction_all(sub)
                                                                 for sub in submissions_all])

df_all = pd.DataFrame({'y_true_train': y_true_train, 'y_pred_train': y_pred_train,
                       'y_true_test': y_true_test, 'y_pred_test': y_pred_test},
                                         index=submissions_all)
df_all.dropna(inplace=True)
df_all.index = df_all.index.str.split('_', n=1, expand=True)

auc_all = []

for idx, serie in df_all.iterrows():
    auc_all += [compute_roc_auc_score(*serie)]
df_all_auc = pd.DataFrame(auc_all, columns=['Public set', 'Private set'], index=df_all.index)
df_roc_auc = df_all_auc.sort_values('Private set', ascending=False).reset_index().drop_duplicates('level_0').set_index(['level_0', 'level_1'])

df = df_roc_auc.reset_index(1).drop('level_1', axis=1).sort_values(by=['Private set'], ascending=True)

jitter = 0.03
df_y_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
df_y_jitter += np.arange(len(df.columns))


fig, ax = plt.subplots(figsize=(8, 4))

g1 = sns.violinplot(data=df.stack().reset_index().rename(columns={0: 'ROC AUC'}), x='ROC AUC', y='level_1', inner=None, legend=None)
# g1.set(xlabel=None)
g1.set(ylabel=None)

for col in df:
    ax.scatter(df[col], df_y_jitter[col],
    alpha=np.where(df.index.isin(best_submissions), 1,np.where(df.iloc[:, 0] > .85, 1, 1)),
    marker='o', # alpha=.40, zorder=1, ms=8, mew=1,
    edgecolors=np.where(df.index.isin(best_submissions), 'black','white'), 
    c=np.where(df.index.isin(best_submissions), 'white',np.where(df.iloc[:, 0] > .85, 'black', 'grey')),
    s=100, zorder=2, linewidth=0.5)

ax.set_yticks(range(len(df.columns)))
ax.set_yticklabels(df.columns)
ax.set_ylim(-0.5,len(df.columns)-0.5)

for idx in np.arange(df.shape[0]):
    ax.plot(df.iloc[idx, [0, 1]], df_y_jitter.iloc[idx, [0, 1]], color = 'grey', linewidth = 0.5, linestyle = '-', zorder=1)

plt.gca().invert_yaxis()
plt.savefig('../figures/fig_5_comparison_public-private.svg', bbox_inches="tight")
