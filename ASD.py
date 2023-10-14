import warnings

import numpy as np
warnings.filterwarnings("ignore", message="Creating an ndarray from ragged nested sequences")
warnings.filterwarnings("ignore", category = FutureWarning)

import pandas as pd

import importlib as il

from problem import get_cv

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

import seaborn as sns

import matplotlib.pyplot as plt

import os as os

import csv as cs


def load_data():
  #Load the data
  from problem import get_train_data
  from problem import get_test_data

  data_train, labels_train = get_train_data()
  data_test, labels_test = get_test_data()
  return data_train, labels_train, data_test, labels_test

def print_gender_info(data_train, data_test):
  #print gender data
  print("Training Data Gender Data")
  print(data_train["participants_sex"].value_counts())

  print("Test Data Gender Data")
  print(data_test["participants_sex"].value_counts())

def evaluation_predict(X,y, Classifier, FeatureExtractor):
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  # Note: in the cross_validate function, they use StratifiedShuffleSplit which allows for resampling
  pipe = make_pipeline(FeatureExtractor(), Classifier())
  cv_custom = StratifiedKFold(n_splits=5, shuffle = True, random_state=42) 

  return cross_val_predict(pipe, X, y, cv=cv_custom, verbose=1, n_jobs=2, method='predict')

def gender_ratio_per_fold(data_train, labels_train):
  #gender ratio per cross-validation fold
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
  data_train_sex = np.array(data_train['participants_sex'])
  fold_number = 1
  for train_index, test_index in cv.split(data_train, labels_train):
      train = data_train_sex[train_index]
      test = data_train_sex[test_index]

      train_male_count = np.count_nonzero(train == 'M')
      train_female_count = np.count_nonzero(train == 'F')

      test_male_count = np.count_nonzero(test == 'M')
      test_female_count = np.count_nonzero(test == 'F')

      print("Fold ", fold_number, ": Training Gender Proportion ", "Female 1 : Male ", round(train_male_count/train_female_count, 2), sep = "")
      print("Fold ", fold_number, ": Test Gender Proportion ", "Female 1 : Male ", round(test_male_count/test_female_count, 2), sep = "")
      fold_number += 1

def load_submission(name):
  classifier_module = il.import_module("submissions."+name+".classifier")
  FeatureExtractor_module = il.import_module("submissions."+name+".feature_extractor")
  return classifier_module.Classifier(), FeatureExtractor_module.FeatureExtractor()

def download_data():
  # Make sure you download the functional data, if it is not already stored on your drive
  from download_data import fetch_fmri_time_series
  fetch_fmri_time_series(atlas='all')

def initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor):
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  return evaluation_predict(data_train, labels_train, Classifier, FeatureExtractor)

def check_for_saved_file(seed):
  return os.path.exists("saved_outcomes/"+str(seed)+".txt")

def check_for_test_dataset(seed):
  try:
    os.mkdir("datasets/"+str(seed))
  except OSError as error:
    # print(error, "yes")
    return True
  return False
  

def save_predictions(seed, predictions):
  warnings.filterwarnings("ignore", message=".*`np.*` is a deprecated alias.*")
  predictions.to_csv("saved_outcomes/"+str(seed)+".txt", index = True)

def save_datasets(seed, new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test):
  warnings.filterwarnings("ignore", message=".*`np.*` is a deprecated alias.*")
  new_train_dataset.to_csv("datasets/" + str(seed) + "/train_dataset_" + str(seed) + ".txt", index = True)
  new_test_dataset.to_csv("datasets/" + str(seed) + "/test_dataset_" + str(seed) + ".txt", index = True)

  # print(new_train_labels)
  with open("datasets/" + str(seed) + "/train_labels_" + str(seed) + ".txt", 'w') as filehandle:
    filehandle.writelines(f"{train_labels}\n" for train_labels in new_train_labels)

  with open("datasets/" + str(seed) + "/test_labels_" + str(seed) + ".txt", 'w') as filehandle:
    filehandle.writelines(f"{test_labels}\n" for test_labels in new_test_labels)

  # with open("datasets/" + str(seed) + "/sex_test_" + str(seed) + ".txt", 'w') as filehandle:
  #   for sex_test_index in sex_test:
  #     filehandle.writelines(f"{sex}\n" for sex in sex_test_index) 

  with open("datasets/" + str(seed) + "/sex_test_" + str(seed) + ".txt", 'w') as filehandle:
      filehandle.writelines(f"{sex}\n" for sex in sex_test) 


def load_predictions(seed):
  return pd.read_csv("saved_outcomes/"+str(seed)+".txt")

def load_datasets(seed):
  train_dataset = pd.read_csv("datasets/" + str(seed) + "/train_dataset_"+str(seed)+".txt")
  test_dataset = pd.read_csv("datasets/" + str(seed) + "/test_dataset_"+str(seed)+".txt")
  # print(train_dataset)
  train_dataset.set_index("subject_id", inplace=True)
  test_dataset.set_index("subject_id", inplace=True)

  train_labels = []
  test_labels = []
  sex_test = []

  with open("datasets/" + str(seed) + "/train_labels_" + str(seed) + ".txt", 'r') as filehandle:

    for line in filehandle:
      line = line.strip()
      train_label = int(line)
      train_labels.append(train_label)

  # print(train_labels)

  with open("datasets/" + str(seed) + "/test_labels_" + str(seed) + ".txt", 'r') as filehandle:

    for line in filehandle:
      line = line.strip()
      test_label = int(line)
      test_labels.append(test_label)
 
  with open("datasets/" + str(seed) + "/sex_test_" + str(seed) + ".txt", 'r') as filehandle:
    for line in filehandle:
      line = line.strip()
      values = line.replace('[', '').replace(']', '').split(",")
      sex_row = [int(value) for value in values]
      sex_test.append(np.array(sex_row))

  # with open("datasets/" + str(seed) + "/test_labels_" + str(seed) + ".txt", 'r') as filehandle:
  #   for line in filehandle:
  #     marker = line[:-1]
  #     test_labels.append(marker)

  # with open("datasets/" + str(seed) + "/sex_test_" + str(seed) + ".txt", 'r') as filehandle:
  #   for line in filehandle:
  #     marker = line[:-1]
  #     sex_test.append(marker)

  return train_dataset, np.array(train_labels), test_dataset, np.array(test_labels), np.array(sex_test)
  

def train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor):
  #Create crossvalidation code
  # folds = 1
  fold_results = []
  cv_custom = StratifiedKFold(n_splits=5, shuffle = True, random_state=42) 

  for train, test in cv_custom.split(data_train, labels_train):


    dataframe_indices = data_train.index.values.copy()
    train_dataset = pd.DataFrame(index = dataframe_indices[train]) #initialise dataframe with key values with unique acquisition

    train_dataset.loc[:, data_train.columns] = data_train.loc[dataframe_indices[train]] #Copy all dataframe information w.r.t key values
    train_dataset.index.name = "subject_id"
    # print(type(train))
    train_labels = labels_train[train]
    
    #Train the model on the full training dataset
    pipe = make_pipeline(FeatureExtractor(), Classifier())

    pipe.fit(train_dataset, train_labels)

    #Test the model on the external dataset (data_test)
    predictions_external = np.round(pipe.predict(data_test))

    fold_results.append(predictions_external)

    # Evaluate the model's performance on the external dataset
    # accuracy_external = accuracy_score(labels_test, predictions_external)
    # print("External Dataset Accuracy of Fold:", folds, "at", round(accuracy_external*100, 2), "%.")
    # folds += 1
  return fold_results

def create_dataframe(name, fold_stat_results):

  dataframe_names = [name + "-overall", name + "-male", name + "-female"]
  dataframe_contents = {"submission": dataframe_names}

  fold_number = 1
  for fold in fold_stat_results:
    TP = []
    FP = []
    FN = []
    TN = []
    AUC = []
    for results in fold:
      print("Fold results", results)
      TP.append(results[0])
      FP.append(results[1])
      FN.append(results[2])
      TN.append(results[3])
      AUC.append(results[4])
    dataframe_contents["TP"+"_"+str(fold_number)] = TP.copy()
    dataframe_contents["FP"+"_"+str(fold_number)] = FP.copy()
    dataframe_contents["FN"+"_"+str(fold_number)] = FN.copy()
    dataframe_contents["TN"+"_"+str(fold_number)] = TN.copy()
    dataframe_contents["AUC"+"_"+str(fold_number)] = AUC.copy()
    fold_number += 1
  # print(dataframe_contents)
  dataframed_results = pd.DataFrame(dataframe_contents)
  dataframed_results.set_index("submission", inplace=True)
  return dataframed_results

def process_results(raw_results, labels, sex):
  result = []
  for predicted_results in raw_results:
    result.append(determine_statistics(labels, predicted_results, sex))
  return result
      
def determine_statistics(labels, results, sex):     
  male_results = results[sex[0]]
  male_labels = labels[sex[0]]
  female_results = results[sex[1]]
  female_labels = labels[sex[1]]

  overall_true_positive = 0
  overall_false_positive = 0
  overall_true_negative = 0
  overall_false_negative = 0

  male_true_positive = 0
  male_false_positive = 0
  male_true_negative = 0
  male_false_negative = 0

  female_true_positive = 0
  female_false_positive = 0
  female_true_negative = 0
  female_false_negative = 0

  i = 0
  while i < len(labels):
    if labels[i] == 1 and results[i] == 1:
      overall_true_positive += 1
    elif results[i] == 0 and labels[i] == 1:
      overall_false_negative += 1
    elif results[i] == 1 and labels[i] == 0:
      overall_false_positive += 1
    elif labels[i] == 0 and results[i] == 0:
      overall_true_negative += 1
    i += 1
  i = 0
  while i < len(male_labels):
    if male_labels[i] == 1 and male_results[i] == 1:
      male_true_positive += 1
    elif male_results[i] == 0 and male_labels[i] == 1:
      male_false_negative += 1
    elif male_results[i] == 1 and male_labels[i] == 0:
      male_false_positive += 1
    elif male_labels[i] == 0 and male_results[i] == 0:
      male_true_negative += 1
    i += 1
  i = 0
  while i < len(female_labels):
    if female_labels[i] == 1 and female_results[i] == 1:
      female_true_positive += 1
    elif female_results[i] == 0 and female_labels[i] == 1:
      female_false_negative += 1
    elif female_results[i] == 1 and female_labels[i] == 0:
      female_false_positive += 1
    elif female_labels[i] == 0 and female_results[i] == 0:
      female_true_negative += 1
    i += 1

  overall_auc = metrics.roc_auc_score(labels, results)
  male_auc = metrics.roc_auc_score(male_labels, male_results)
  female_auc = metrics.roc_auc_score(female_labels, female_results)

  return [(overall_true_positive, overall_false_positive, overall_false_negative, overall_true_negative, overall_auc), 
          (male_true_positive, male_false_positive, male_false_negative, male_true_negative, male_auc), 
          (female_true_positive, female_false_positive, female_false_negative, female_true_negative, female_auc)]

def auc_roc(results):
  # print(type(results), results)
  test_names = results.index.values.tolist()
  headings = results.columns.values.tolist()

  auc = []
  names = []

  i = 1
  while ("AUC"+"_"+str(i)) in headings:
    AUC_fold = results["AUC"+"_"+str(i)].values.tolist()

    j = 0
    while j < len(AUC_fold):
      auc.append(AUC_fold[j])
      j += 1

    k = 0
    while k < len(test_names):
      names.append(test_names[k]+"-"+str(i))
      k += 1
    i += 1
  consolidated_test_names = []
  auc_sex = []
  sex = []

  l = 0
  while l < len(names):
    split_names = str.split(names[l], "-")
    test_name = split_names[0]
    if split_names[1] != "overall":
      sex.append(split_names[1])
      consolidated_test_names.append(test_name)
      auc_sex.append(auc[l]*100)

    l += 1
  
  m = 0
  while m < len(auc):
    auc[m] *= 100
    m += 1
  
  auc_df = pd.DataFrame({"Submissions": consolidated_test_names,
                        "Results": auc_sex,
                        "Sex": sex})
  return auc_df

def plot_auc_df(auc_df, seed):
  sns.set_theme(style="whitegrid")
  plot = sns.violinplot(data=auc_df, x="Submissions", y="Results", split=True, hue="Sex", inner="stick")
  plot.set_title('AUC-ROC Performance of 10 Best Submissions: Seed '+str(seed))
  plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)  
  plot.set_xlabel('Submissions')
  plot.set_ylabel('AUC (%)')
  plot.legend(loc=(1.040, 0.5))
  plt.savefig('results/Seeded Results/'+str(seed)+'/auc_roc.png', bbox_inches="tight")
  plt.show()

def general_accuracy(results):

  test_names = results.index.values.tolist()
  headings = results.columns.values.tolist()

  ga = []
  names = []

  i = 1
  while ("TP"+"_"+str(i)) in headings:
    TP_fold = results["TP"+"_"+str(i)].values.tolist()
    FP_fold = results["FP"+"_"+str(i)].values.tolist()
    FN_fold = results["FN"+"_"+str(i)].values.tolist()
    TN_fold = results["TN"+"_"+str(i)].values.tolist()

    j = 0
    while j < len(TP_fold):
      ga.append((TP_fold[j] + TN_fold[j]) / (TP_fold[j] + FP_fold[j] + FN_fold[j] + TN_fold[j]))
      j += 1

    k = 0
    while k < len(test_names):
      names.append(test_names[k]+"-"+str(i))
      k += 1
    i += 1

  consolidated_test_names = []
  ga_sex = []
  sex = []

  l = 0
  while l <len(names):
    split_names = str.split(names[l], "-")
    test_name = split_names[0]
    if split_names[1] != "overall":
      sex.append(split_names[1])
      consolidated_test_names.append(test_name)
      ga_sex.append(ga[l]*100)
    l += 1

  ga_df = pd.DataFrame({"Submissions": consolidated_test_names,
                        "Results": ga_sex,
                        "Sex": sex
                        })

  return ga_df

def plot_ga_df(ga_df, seed):
  sns.set_theme(style="whitegrid")
  plot = sns.violinplot(data=ga_df, x="Submissions", y="Results",  split = True, hue="Sex", inner="stick")
  plot.set_title('General Accuracy  Performance of 10 Best Submissions: Seed '+str(seed))
  plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)
  plot.set_xlabel('Submissions')
  plot.set_ylabel('Accuracy (%)')
  plot.legend(loc=(1.040, 0.5))
  plt.savefig('results/Seeded Results/'+str(seed)+'/ga.png', bbox_inches="tight")
  plt.show()

def equalised_odds(results):

  test_names = results.index.values.tolist()
  headings = results.columns.values.tolist()

  #tpr
  tpr = []
  names_tpr = []  
  
  i = 1
  while ("TP"+"_"+str(i)) in headings:
    TP_fold = results["TP"+"_"+str(i)].values.tolist()
    FN_fold = results["FN"+"_"+str(i)].values.tolist()

    j = 0
    while j < len(TP_fold):
      tpr.append(TP_fold[j]/(TP_fold[j] + FN_fold[j]))
      j += 1

    k = 0
    while k < len(test_names):
      names_tpr.append(test_names[k]+"-"+str(i))
      k += 1
    i += 1

  consolidated_test_names_tpr = []
  eo_tpr = []
  eo_tpr_fpr = []

  l = 0
  while l <len(names_tpr):
    split_names = str.split(names_tpr[l], "-")
    test_name = split_names[0]+"-"+split_names[2]
    if split_names[1] != "overall":
      if test_name not in consolidated_test_names_tpr:
        consolidated_test_names_tpr.append(test_name)
        eo_tpr.append(tpr[l])
        eo_tpr_fpr.append("tpr")
      else:
        eo_tpr[consolidated_test_names_tpr.index(test_name)] -= tpr[l]
    l += 1
  
  #fpr
  fpr = []
  names_fpr = []  
  
  i = 1
  while ("TP"+"_"+str(i)) in headings:
    FP_fold = results["FP"+"_"+str(i)].values.tolist()
    TN_fold = results["TN"+"_"+str(i)].values.tolist()

    j = 0
    while j < len(FP_fold):
      fpr.append(FP_fold[j]/(FP_fold[j] + TN_fold[j]))
      j += 1

    k = 0
    while k < len(test_names):
      names_fpr.append(test_names[k]+"-"+str(i))
      k += 1
    i += 1

  consolidated_test_names_fpr = []
  eo_fpr = []

  l = 0
  while l <len(names_fpr):
    split_names = str.split(names_fpr[l], "-")
    test_name = split_names[0]+"-"+split_names[2]
    if split_names[1] != "overall":
      if test_name not in consolidated_test_names_fpr:
        consolidated_test_names_fpr.append(test_name)
        eo_fpr.append(fpr[l])
        eo_tpr_fpr.append("fpr")
      else:
        eo_fpr[consolidated_test_names_fpr.index(test_name)] -= fpr[l]
    l += 1

  generalised_submission_names = []
  m = 0

  while m < len(eo_tpr):
    eo_tpr[m] *= 100

    consolidated_test_names_split = str.split(consolidated_test_names_tpr[m], "-")
    generalised_submission_names.append(consolidated_test_names_split[0])
    m += 1

  m = 0
  while m < len(eo_fpr):
    eo_fpr[m] *= 100

    consolidated_test_names_split = str.split(consolidated_test_names_fpr[m], "-")
    generalised_submission_names.append(consolidated_test_names_split[0])
    m += 1

  # print(len(generalised_submission_names), len(np.concatenate([eo_tpr, eo_fpr])), len(eo_tpr_fpr))
  
  eo_df = pd.DataFrame({"Submissions": generalised_submission_names,
                        "Results": np.concatenate([eo_tpr, eo_fpr]), 
                        "TPR:FPR": eo_tpr_fpr})

  return eo_df
  # print(eo_df)


def plot_eo_df(eo_df, seed):
  sns.set_theme(style="whitegrid")
  plot = sns.violinplot(data=eo_df, x="Submissions", y="Results",  split = True, hue = "TPR:FPR", inner="stick")
  plot.set_title('Equalised Odds Results Performance of 10 Best Submissions: Seed '+str(seed))
  plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)  
  plot.set_xlabel('Submissions')
  plot.set_ylabel('True Positive Rate/False Positive Rate (%)')
  plt.savefig('results/Seeded Results/'+str(seed)+'/eo.png', bbox_inches="tight")
  # plot.legend(loc=(1.040, 0.5))
  plt.show()


def separate_test_suite(overall_set, overall_labels):
  # print(overall_labels.size)
  #Gather all indices which are unique. (w.r.t site, sex and neurostatus)
  test_indices = determine_test_sample_indices(overall_set, overall_labels)
  train_indices = np.delete(np.arange(overall_labels.size), test_indices)

  subject_id_test = overall_set.index.values.copy() #Key_values of original dataframe
  subject_id_train = overall_set.index.values.copy() #Key_values of original dataframe

  #New Dataframe Transfer of testing data
  test_dataset = pd.DataFrame(index = subject_id_test[test_indices]) #initialise dataframe with key values with unique acquisition
  test_dataset.loc[:, overall_set.columns] = overall_set.loc[subject_id_test[test_indices]] #Copy all dataframe information w.r.t key values
  test_dataset.index.name = "subject_id"

  #New Dataframe Transfer of training data
  train_dataset = pd.DataFrame(index = subject_id_train[train_indices]) #initialise dataframe with key values with unique acquisition
  train_dataset.loc[:, overall_set.columns] = overall_set.loc[subject_id_train[train_indices]] #Copy all dataframe information w.r.t key values
  train_dataset.index.name = "subject_id"
  #Discern validity of sample. (Acquisition site has neurodiverse/neurotypical Male/Female)
  # print(test_dataset.loc[:,["participants_site", "participants_sex"]].value_counts())
  # print(train_dataset.info())
  # print(test_dataset.info())
  return train_dataset, overall_labels[train_indices], test_dataset, overall_labels[test_indices]

def determine_test_sample_indices(overall_set, overall_labels):

  test_data = []
  i = 0

  site_data = overall_set["participants_site"].values
  sex_data = overall_set["participants_sex"].values
  y = overall_labels.copy()

  #Iterate over all unique values from site, sex and 'neurostatus'
  for site_i in np.unique(site_data): #Iterate over array of unique sites [1, 2, ..., 34]
    for sex_i in np.unique(sex_data): #Iterate over aray of unique sexes ['M', 'F']
      for y_i in np.unique(y): #Iterate over array of unique labels ['0', '1']

        #Determine all indicies of data satisfy unique values being iterated over
        neuro_label = np.where(y.reshape(-1) == y_i)[0] #List of indices that equal y_i
        sex_label = np.where(sex_data.reshape(-1) == sex_i)[0] #List of indices that equal sex_i
        site_label = np.where(site_data.reshape(-1) == site_i)[0] #List of indices that equal site_i

        #Determine which indices satisfy all three above conditions
        intersection = np.intersect1d(np.intersect1d(neuro_label, sex_label), site_label)

        if intersection.shape[0] > 0: #If there is at least one scenario which satisfies all three conditions
          test_data.append(np.random.choice(intersection)) #Choose a random index
          i += 1 #Count the total unique scenarios

  test_data = np.array(sorted(test_data)) #sort the resultant array
  # print(i)
  return(test_data)

#As the function name describes, joining the original training and test dataframes together.
#Furthermore, joins the labelled outcome arrays together to maintain tracking of subject to result
#Please note both solutions preserve the original order of the two separate datasets and labels
#Finally, these have been concatenated with the resultant order: Original training dataset -> Original test dataset
def join_original_datasets(data_train, labels_train, data_test, labels_test):
  return pd.concat([data_train, data_test], ignore_index=False), np.concatenate((labels_train, labels_test))

#Function checks each index of 'dataset_one' and checks if it is 'in' 'dataset_two'.
#If there are no instances where this occurs, the function will return true (therefore having unique indices)
#If there are instances of two datasets with duplicate indices, the function will return false at the first instance.
def determine_unique_dataframe(dataset_one, dataset_two):
  dataset_one_indices = dataset_one.index.values.copy()
  dataset_two_indices = dataset_two.index.values.copy()
  for index in dataset_one_indices:
    if index in dataset_two_indices:
      return False
  return True

def sex_index_split(test_dataset):
  male_indices = []
  female_indices = []

  sex_indices = test_dataset["participants_sex"].to_numpy()
  i = 0
  while i < len(sex_indices):
    if sex_indices[i] == 'M':
      male_indices.append(i)
    else:
      female_indices.append(i)
    i += 1

  return (male_indices, female_indices)

def organise_results(name, raw_submission_results):
  # print(raw_submission_results)
  # name = "hi"
  summed_result = {}
  summed_result[name] = {}
  folds = 0
  for test_name in raw_submission_results:
    summed_result[name][test_name] = {}
    folds = 0
    for fold_number in raw_submission_results[test_name]:
      folds+=1
      for test_category_name in raw_submission_results[test_name][fold_number]:
        if test_category_name not in summed_result[name][test_name]:
          summed_result[name][test_name][test_category_name] = {}
        if "Average" in summed_result[name][test_name][test_category_name]:
          summed_result[name][test_name][test_category_name]["Average"] = raw_submission_results[test_name][fold_number][test_category_name] + summed_result[name][test_name][test_category_name]["Average"]
        else:
          summed_result[name][test_name][test_category_name]["Average"] = raw_submission_results[test_name][fold_number][test_category_name]
        summed_result[name][test_name][test_category_name][fold_number] = raw_submission_results[test_name][fold_number][test_category_name]
        # print(test_name, test_category_name, fold_number, ":", summed_result[name][test_name][test_category_name][fold_number])

  # print(summed_result[name])
  for test_name in summed_result[name]:
    for test_category_name in summed_result[name][test_name]:
      summed_result[name][test_name][test_category_name]["Average"] = summed_result[name][test_name][test_category_name]["Average"]/folds
      # print(test_name, test_category_name, summed_result[test_name][test_category_name]/folds)
  # print(folds, averaged_results)

  organised_results = pd.DataFrame.from_dict({(i, j, k): summed_result[i][j][k]
                                              for i in summed_result.keys()
                                              for j in summed_result[i].keys()
                                              for k in summed_result[i][j].keys()}, orient = 'index')
  organised_results.index.set_names(['submission', 'type', 'category'])
  # organised_results.columns = ["Average Results"]
  # print(organised_results.info())
  # print(organised_results.index)
  # print(organised_results)
  return organised_results

#Functions to run each submission
def run_pearrr_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "pearrr_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.pearrr_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.pearrr_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)

  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  fold_stat_results = process_results(fold_results, labels_test, sex_test)

  return create_dataframe(name, fold_stat_results)

def run_abethe_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "abethe_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.abethe_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.abethe_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)

  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  fold_stat_results = process_results(fold_results, labels_test, sex_test)

  return create_dataframe(name, fold_stat_results)

def run_amicie_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "amicie_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.amicie_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.amicie_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)

  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  fold_stat_results = process_results(fold_results, labels_test, sex_test)

  return create_dataframe(name, fold_stat_results)

def run_ayoub_ghriss_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "ayoub_ghriss_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.ayoub_ghriss_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.ayoub_ghriss_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)

  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  fold_stat_results = process_results(fold_results, labels_test, sex_test)

  return create_dataframe(name, fold_stat_results)

def run_lbg_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "lbg_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.lbg_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.lbg_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)

  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  fold_stat_results = process_results(fold_results, labels_test, sex_test)

  return create_dataframe(name, fold_stat_results)

def run_mk_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "mk_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.mk_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.mk_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)

  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  fold_stat_results = process_results(fold_results, labels_test, sex_test)

  return create_dataframe(name, fold_stat_results)

def run_nguigui_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "nguigui_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.nguigui_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.nguigui_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)

  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  fold_stat_results = process_results(fold_results, labels_test, sex_test)

  return create_dataframe(name, fold_stat_results)

def run_Slasnista_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "Slasnista_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.Slasnista_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.Slasnista_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)

  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  fold_stat_results = process_results(fold_results, labels_test, sex_test)

  return create_dataframe(name, fold_stat_results)

def run_vzantedeschi_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "vzantedeschi_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.vzantedeschi_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.vzantedeschi_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)

  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  fold_stat_results = process_results(fold_results, labels_test, sex_test)

  return create_dataframe(name, fold_stat_results)

def run_wwwwmmmm_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "wwwwmmmm_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.wwwwmmmm_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.wwwwmmmm_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)

  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  fold_stat_results = process_results(fold_results, labels_test, sex_test)

  return create_dataframe(name, fold_stat_results)

def develop_test_set(seed):
  # print(check_for_test_dataset(seed))
  if check_for_test_dataset(seed) == False:
    #Load data
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    data_train, labels_train, data_test, labels_test = load_data()

    # merge provided training datasets and testing datasets
    merged_dataset, merged_labels = join_original_datasets(data_train, labels_train, data_test, labels_test)

    #Generate randomised test dataset and remove from training dataset.
    new_train_dataset, new_train_labels, new_test_dataset, new_test_labels = separate_test_suite(merged_dataset, merged_labels)

    sex_test = sex_index_split(new_test_dataset)
    save_datasets(seed, new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)
  else:
    new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test = load_datasets(seed)


  return new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test


def run_tests(seed, new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test):
  np.random.default_rng(seed=42)
  
  try:
    os.mkdir("results")
  except OSError as error:
    print("results folder exists already, punk")

  try:
    os.mkdir("results/Seeded Results")
  except OSError as error:
    print("Seeded Results folder exists already, punk")

  try:
    os.mkdir("results/Seeded Results/"+str(seed))
  except OSError as error:
    print("Folder " + str(seed) + " exists already, punk")



  #Display initial dataset information
  # print(data_train.info())
  # print(labels_train.size)
  # print(data_test.info())
  # print(labels_test.size)

  #Display information of initial datasets with respect to acquisition sites.
  # print(data_train["participants_site"].value_counts().sort_index())
  # print(data_test["participants_site"].value_counts().sort_index())

  #Merge both intial datasets to preserve all datasets


  # check_state =  np.random.get_state()
  # check_seed = check_state[1][0]
  # check_seed = np.random.get_state()
  # print("Yo, the test set was developed with random seed: ", check_seed)

  # print(sex_test)
  #Check uniqueness of training and test datasets
  # print(determine_unique_dataframe(new_train_dataset, new_test_dataset))
  # print(determine_unique_dataframe(new_train_dataset, new_train_dataset))

  #Display information of training/testing datasets and their results.
  # print(data_train.index)
  # print(new_train_dataset.index)
  # print(new_train_dataset.info())
  # print(new_train_labels.size)
  # print(new_test_dataset.info())
  # print(new_test_labels.size)

  #Display training/testing results
  # print(labels_train)
  # print(labels_test)
  # print(merged_labels)

  #Indexing a dataframe
  # print(merged_dataset.loc[10631804530197433027])

  #Display gender ratio information
  # print_gender_info()
  # gender_ratio_per_fold()
  submissions = []

 

  if check_for_saved_file(seed) == False:
    #Train and test submissions
    np.random.default_rng(seed=seed)
    
    submissions = run_pearrr_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)
    submissions = pd.concat([submissions, run_abethe_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    submissions = pd.concat([submissions, run_amicie_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    submissions = pd.concat([submissions, run_ayoub_ghriss_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    submissions = pd.concat([submissions, run_lbg_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    submissions = pd.concat([submissions, run_mk_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    # submissions = run_nguigui_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)
    submissions = pd.concat([submissions, run_nguigui_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    submissions = pd.concat([submissions, run_Slasnista_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    submissions = pd.concat([submissions, run_vzantedeschi_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    submissions = pd.concat([submissions, run_wwwwmmmm_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    save_predictions(seed, submissions)
  else:
    submissions = load_predictions(seed)
    submissions.set_index("submission", inplace=True)
  # for a in submissions:
  #   for b in a:
  #     # for c in b:
  #     print(b.type)
  # print(submissions)
  # create_violin_graph(submissions)
  # fig= plt.figure(figsize=(90, 10))
  # submissions["Average"].plot.bar()
  # plt.xticks(rotation = 90)
  # plt.legend(loc=(1.04, 0))
  # plt.show()
  auc_results = auc_roc(submissions)
  ga_results = general_accuracy(submissions)
  eo_results = equalised_odds(submissions)
  # plot_auc_df(auc_results, seed)
  # plot_ga_df(ga_results, seed)
  # plot_eo_df(eo_results, seed)

  # auc_results['seed'] = seed
  # ga_results['seed'] = seed
  # eo_results['seed'] = seed

  return auc_results, ga_results, eo_results
  
def create_ridgeline(seeds):
  seed_auc = []
  seed_ga = []
  seed_eo = []

  new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test = develop_test_set(42)

  #collect all the test results for each auc, ga and eo.
  for seed in seeds:
    new_tests = run_tests(seed, new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)
    seed_auc.append(new_tests[0])
    seed_ga.append(new_tests[1])
    seed_eo.append(new_tests[2])

  # print(seed_auc[0]) #should be the first auc results of the first seed.
  # print(seed_auc)
  i = 0
  merged_seed_auc = seed_auc[i]
  merged_seed_ga = seed_ga[i]
  merged_seed_eo = seed_eo[i]
  while i < len(seeds)-1:
    merged_seed_auc = pd.concat([merged_seed_auc, seed_auc[i+1]])
    merged_seed_auc = pd.concat([merged_seed_ga, seed_ga[i+1]])
    merged_seed_auc = pd.concat([merged_seed_eo, seed_eo[i+1]])
    i+=1
  # print(merged_seed_auc) #Should be len(seeds)*

  # plt.figure(figsize = (8, 5))

  # categories = merged_seed_auc['Submissions'].unique()

  # for category in categories:
  #   subset = merged_seed_auc[merged_seed_auc['Submissions'] == category]
  #   sns.kdeplot(data=subset["Results"], label=category, shade = True)

  submission_dict = {}
  submission_names = seed_auc[0]["Submissions"].unique()
  # print(submission_names)
  i = 0
  for name in submission_names:
    submission_dict[i] = name
    i+=1
  # print(submission_dict)

  sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}) 
  # we generate a color palette with Seaborn.color_palette()
  pal = sns.cubehelix_palette(10, rot=-.25, light=.7)

  # in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
  # g = sns.FacetGrid(merged_seed_auc, row='Submissions', aspect = 15, height = 1, palette = pal)
  g = sns.FacetGrid(merged_seed_eo, row="Submissions", hue="Submissions", aspect=15, height=1, palette=pal)
  
  # then we add the densities kdeplots for each month
  # g.map(sns.kdeplot, 'Results', bw_adjust = 1, clip_on=False, fill=True, alpha=1, linewidth=1.5) 
  g.map(sns.kdeplot, "Results", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)

  # here we add a white line that represents the contour of each kdeplot
  # g.map(sns.kdeplot, 'Results', bw_adjust=1, clip_on=False, color="w", lw=2)
  
  # here we add a horizontal line for each plot
  # g.map(plt.axhline, y=0, lw=2, clip_on=False)
  g.map(sns.kdeplot, "Results", clip_on=False, color="w", lw=2, bw_adjust=.5)


  g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
  # we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
  # notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
  # i = 0
  # g.axes[0].set_ylabel("")
  # for i, ax in enumerate(g.axes.flat):
  #   ax.text(0, 0.02, submission_names[i], fontweight='bold', fontsize=15, color=ax.lines[-1].get_color())
  #   ax.set_ylabel("")
  #   plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')

  # we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
  # g.fig.subplots_adjust(hspace=-0.3)

  # # eventually we remove axes titles, yticks and spines
  # g.set_titles("")
  # g.set(yticks=[])
  # g.despine(bottom=True, left=True)
  # g.set_ylabels("")
  # # i=0
  # for i, ax in enumerate(g.axes.flat):
  #     ax.text(15, 0.02, submission_names[i], fontweight='bold', fontsize=15, color=ax.lines[-1].get_color())



  # plt.xlabel('Performance (%)', fontweight='bold', fontsize=15)
  # g.fig.suptitle('AUC-ROC Performance of each submission across all seeds',
  #               ha='center',
  #               fontsize=20,
  #               fontweight=20)

  # plt.show()
  def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)
    
  g.map(label, "Submissions")
  
  g.figure.subplots_adjust(hspace=-.25)
  
  # Remove axes details that don't play well with overlap
  g.set_titles("")
  g.set(yticks=[], ylabel="")
  g.despine(bottom=True, left=True)


  
def test_suite():
  x = np.random.rand()
  # x = 1
  results = {}
  ga_folds = {}
  eo_folds = {}

  ga_tests_1 = {}
  ga_tests_1['overall'] = 0.6*x
  ga_tests_1['male'] = 0.6060606060606061*x
  ga_tests_1['female'] = 0.5918367346938775*x
  ga_folds[1] = ga_tests_1

  ga_tests_2 = {}
  ga_tests_2['overall'] = 0.6347826086956522*x
  ga_tests_2['male'] = 0.6363636363636364*x
  ga_tests_2['female'] = 0.6326530612244898*x
  ga_folds[2] = ga_tests_2

  ga_tests_3 = {}
  ga_tests_3['overall'] = 0.6347826086956522*x
  ga_tests_3['male'] = 0.6363636363636364*x
  ga_tests_3['female'] = 0.6326530612244898*x
  ga_folds[3] = ga_tests_3

  ga_tests_4 = {}
  ga_tests_4['overall'] = 0.6260869565217392*x
  ga_tests_4['male'] = 0.6060606060606061*x
  ga_tests_4['female'] = 0.6530612244897959*x
  ga_folds[4] = ga_tests_4

  ga_tests_5 = {}
  ga_tests_5['overall'] = 0.6*x
  ga_tests_5['male'] = 0.6212121212121212*x
  ga_tests_5['female'] = 0.5714285714285714*x
  ga_folds[5] = ga_tests_5

  eo_tests_1 = {}
  eo_tests_1['overall'] = 0.17647058823529416*x
  eo_folds[1] = eo_tests_1

  eo_tests_2 = {}
  eo_tests_2['overall'] = 0.1642156862745099*x
  eo_folds[2] = eo_tests_2
  
  eo_tests_3 = {}
  eo_tests_3['overall'] = 0.08088235294117652*x
  eo_folds[3] = eo_tests_3
  
  eo_tests_4 = {}
  eo_tests_4['overall'] = 0.20588235294117652*x
  eo_folds[4] = eo_tests_4
  
  eo_tests_5 = {}
  eo_tests_5['overall'] = 0.15931372549019612*x
  eo_folds[5] = eo_tests_5

  results['ga'] = ga_folds
  results['eo'] = eo_folds
  return results

# group_results = {}
# results_1 = test_suite()
# results_2 = test_suite()
# # print(results_1)

# organised_1 = organise_results("test_1", results_1)
# # print(organised_1)
# organised_2 = organise_results("test_2", results_2)

# group_results = pd.concat([organised_1, organised_2])

# # group_results["first"] = organised_1
# # group_results["second"] = organised_2

# # print(group_results.info())
# # print(group_results.index)
# print(group_results)

# create_violin_graph(group_results)
# fig= plt.figure(figsize=(30, 10))
# # keys = group_results["Average"].index
# # values = group_results["Average"]
# group_results["Average"].plot.bar()
# plt.xticks(rotation = 90)
# plt.legend(loc=(1.04, 0))
# plt.show()

# run_tests(12)

seeds = [11]
# seeds = [11, 11*11, 11*11*11, 11*11*11*11, 11*11*11*11*11]
create_ridgeline(seeds)

# new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test = develop_test_set(42)
# # print(type(new_train_labels), new_train_labels)
# # print(type(sex_test), sex_test)

# submissions = run_nguigui_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)
# auc_results = auc_roc(submissions)
# ga_results = general_accuracy(submissions)
# eo_results = equalised_odds(submissions)

# plot_auc_df(auc_results, 42)
# plot_ga_df(ga_results, 42)
# plot_eo_df(eo_results, 42)