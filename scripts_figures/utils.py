import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from problem import get_train_data, get_test_data


def _get_data_rdb_out():
    """Split the data to provide the true label and data with only RDB as
    test.

    Returns
    -------
    X_train : ndarray, shape (n_train_samples, )
        The training data without RDB.
    X_test : ndarray, shape (n_test_samples, )
        The testing data corresponding to the RDB subjects.
    y_train : ndarray, shape (n_train _samples, )
        The labels of the training set.
    y_test : ndarrays, shape (n_test_samples, )
        The labels of the testing set.

    """
    rdb_idx = np.load('rdb_idx.npy')
    X_test, y_test = get_test_data('..')
    X_train, y_train = get_train_data('..')
    X_test_idx = X_test.index.values
    X_rdb_idx = [X_test_idx == ii for ii in rdb_idx]
    X_rdb_idx = np.vstack(X_rdb_idx)
    X_rdb_idx = np.sum(X_rdb_idx, axis=0).astype(bool)

    return (pd.concat([X_train, X_test[~X_rdb_idx]], axis=0),
            X_test[X_rdb_idx],
            np.concatenate([y_train, y_test[~X_rdb_idx]]),
            y_test[X_rdb_idx])


def load_train_test_prediction(submission_name):
    """Load the true and predicted labels for a given submission.

    Parameters
    ----------
    submission_name : str
        The name of the submission (e.g. 'abethe_anatomy').

    Returns
    -------
    y_true_train : ndarray, shape (n_train_samples, )
        The true labels on the training set.
    y_pred_train : ndarray, shape (n_train_samples, )
        The predicted labels on the training set.
    y_true_test : ndarray, shape (n_test_samples, )
        The true labels on the testing set.
    y_pred_test : ndarray, shape (n_test_samples, )
        The predicted labels on the testing set.

    """
    path_store_pred = os.path.join('../submissions', submission_name,
                                   'training_output')

    y_pred_train = np.loadtxt(os.path.join(path_store_pred, 'y_pred__bagged_valid.csv'))
    y_pred_test = np.loadtxt(os.path.join(path_store_pred, 'y_pred__bagged_test.csv'))

    _, y_true_train = get_train_data('..')
    _, y_true_test = get_test_data('..')

    # remove y_train values for which we have no prediction because they never went in the test part of the cv
    y_true_train = y_true_train[~np.isnan(y_pred_train[:,1])]
    y_pred_train = y_pred_train[~np.isnan(y_pred_train[:,1]), :]

    return (y_true_train, y_pred_train, y_true_test, y_pred_test)

def load_train_test_blended_prediction(group):
    path_store_pred = os.path.join('../submissions', 'training_output_' + group)
    
    # we take foldwise because it corresponds to combined, may evolve with future versions of ramp workflow
    y_pred_train = np.loadtxt(os.path.join(path_store_pred, 'y_pred_foldwise_best_bagged_valid.csv'))
    y_pred_test = np.loadtxt(os.path.join(path_store_pred, 'y_pred_foldwise_best_bagged_test.csv'))
    
    _, y_true_train = get_train_data('..')
    _, y_true_test = get_test_data('..')
   
    # remove y_train values for which we have no prediction because they never went in the test part of the cv
    y_true_train = y_true_train[~np.isnan(y_pred_train[:,1])]
    y_pred_train = y_pred_train[~np.isnan(y_pred_train[:,1]), :]
 
    return (y_true_train, y_pred_train, y_true_test, y_pred_test)

def load_train_test_prediction_aims(submission_name):
    path_store_pred = os.path.join('../..', 'eu-aims', 'submissions', submission_name, 'training_output')
    
    y_pred_train = np.loadtxt(os.path.join(path_store_pred, 'y_pred__bagged_valid.csv'))
    y_pred_test = np.loadtxt(os.path.join(path_store_pred, 'y_pred__bagged_test.csv'))

    _, y_true_train = get_train_data("../../eu-aims")
    _, y_true_test = get_test_data("../../eu-aims")
    
    # remove y_train values for which we have no prediction because they never went in the test part of the cv
    y_true_train = y_true_train[~np.isnan(y_pred_train[:,1])]
    y_pred_train = y_pred_train[~np.isnan(y_pred_train[:,1]), :]
    
    return (y_true_train, y_pred_train, y_true_test, y_pred_test)

def load_train_test_blended_prediction_lc(nsub, it):
    data_dir = os.path.join('../learning_curve', 'it_{}_{}'.format(nsub, it))
    path_store_pred = os.path.join(data_dir, 'submissions/training_output')
    
    if not os.path.exists(os.path.join(path_store_pred, 'y_pred_foldwise_best_bagged_valid.csv')):
        return (None, None, None, None)

    # we take foldwise because it corresponds to combined, may evolve with future versions of ramp workflow
    y_pred_train = np.loadtxt(os.path.join(path_store_pred, 'y_pred_foldwise_best_bagged_valid.csv'))
    y_pred_test = np.loadtxt(os.path.join(path_store_pred, 'y_pred_foldwise_best_bagged_test.csv'))
    
    _, y_true_train = get_train_data(data_dir)
    _, y_true_test = get_test_data(data_dir)
    
    # remove y_train values for which we have no prediction because they never went in the test part of the cv
    y_true_train = y_true_train[~np.isnan(y_pred_train[:,1])]
    y_pred_train = y_pred_train[~np.isnan(y_pred_train[:,1]), :]
    
    return (y_true_train, y_pred_train, y_true_test, y_pred_test)

def compute_roc_auc_score(y_true_train, y_pred_train,
                          y_true_test, y_pred_test):
    """Compute the ROC-AUC for the training and testing set.

    Parameters
    ----------
    y_true_train : ndarray, shape (n_train_samples, )
        The true labels on the training set.
    y_pred_train : ndarray, shape (n_train_samples, )
        The predicted labels on the training set.
    y_true_test : ndarray, shape (n_test_samples, )
        The true labels on the testing set.
    y_pred_test : ndarray, shape (n_test_samples, )
        The predicted labels on the testing set.

    Returns
    -------
    roc_auc_train : float,
        The ROC-AUC on the training set.
    roc_auc_test : float,
        The ROC-AUC on the testing set.

    """
    return (roc_auc_score(y_true_train, y_pred_train[:, 1]),
            roc_auc_score(y_true_test, y_pred_test[:, 1]))
