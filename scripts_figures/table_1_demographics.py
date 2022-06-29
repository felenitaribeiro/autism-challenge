import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

import sys
sys.path.append("..")
from problem import get_train_data, get_test_data

X_train, y_train = get_train_data("..")
X_test, y_test = get_test_data("..")
X_test_aims, y_test_aims = get_test_data("../../eu-aims/")

X_iq = pd.read_csv('../../demographics/participants2.csv').set_index('subject_id')

X_train["type"] = "public set"
X_test["type"] = "private set"
X_test_aims["type"] = "replication set"
# X_train["target"] = y_train
# X_test["target"] = y_test
# X_test_aims["target"] = y_test_aims

X = pd.concat((X_train, X_test, X_test_aims), axis=0)[
    ["participants_site", "participants_sex", "participants_age", "type"]
]
y = np.hstack((y_train, y_test, y_test_aims))
X["target"] = y
X = X.merge(X_iq, left_index=True, right_on='subject_id', how='left')

summary = pd.DataFrame(columns=['dataset', 'variable', 'controls', 'ASD']).set_index(['dataset', 'variable'])

for group, data in X.groupby('type', sort=False):
    for i, data_dx in data.groupby('target'):
        sex_counts = data_dx.participants_sex.value_counts()
        summary.loc[(group, 'Sex'), summary.columns[i]] = (sex_counts.apply(str) + ' ' + sex_counts.index).str.cat(sep=', ')

    age_mean = data.groupby('target').participants_age.mean().round(2).apply(str)
    age_std = data.groupby('target').participants_age.std().round(2).apply(str)
    age_min = data.groupby('target').participants_age.min().round(2).apply(str)
    age_max = data.groupby('target').participants_age.max().round(2).apply(str)
    summary.loc[(group, 'Age (years)'), :] = list(age_mean + ' \u00B1 ' + age_std + ' (' + age_min + ' - ' + age_max + ')')

    iq_mean = data.groupby('target').fsiq.mean().round(2).apply(str)
    iq_std = data.groupby('target').fsiq.std().round(2).apply(str)
    iq_min = data.groupby('target').fsiq.min().round(2).apply(str)
    iq_max = data.groupby('target').fsiq.max().round(2).apply(str)
    summary.loc[(group, 'Full scale IQ'), :] = list(iq_mean + ' \u00B1 ' + iq_std + ' (' + iq_min + ' - ' + iq_max + ')')

summary.to_csv('../figures/table_1_demographics.tsv', sep='\t')
