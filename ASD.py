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
from sklearn. metrics import roc_auc_score
from sklearn. metrics import roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import seaborn as sns

import matplotlib.pyplot as plt

from os.path import exists


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
  return exists("saved_outcomes/"+str(seed)+".txt")

def save_predictions(seed, predictions):
  warnings.filterwarnings("ignore", message=".*`np.*` is a deprecated alias.*")

  f = open("saved_outcomes/"+str(seed)+".txt", "w")
  f.close()
  predictions.to_csv("saved_outcomes/"+str(seed)+".txt", index = True, index_label = ['submission', 'type', 'category'])

def load_predictions(seed):
  return pd.read_csv("saved_outcomes/"+str(seed)+".txt", index_col = ['submission', 'type', 'category'])

def plot_auc(labels_train, predictions, name):
  #define auc-roc score
  auc_roc_score = roc_auc_score(labels_train, predictions)

  #print decimal value
  print("AUC-ROC Score:", auc_roc_score)

  #determine false positive rate and true positive rate
  fpr, tpr, thresholds = roc_curve(labels_train, predictions)

  # Plotting the AUC-ROC curve
  plt.plot(fpr, tpr, label='AUC = %0.3f' % auc_roc_score)
  plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random guessing
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('AUC-ROC Curve for ' + name)
  plt.legend(loc='lower right')
  plt.show()
  return auc_roc_score

def general_accuracy_old(predictions, data_train, labels_train, seed):
  
  # print("General Accurracy: True Positive and True Negative Accuracy")
  # print(predictions)
  # print(cv_split)
  # print(data_train)
  # print(labels_train)

  warnings.filterwarnings("ignore", message=".*`np.*` is a deprecated alias.*")
  cv_ga = StratifiedKFold(n_splits=5, shuffle = True, random_state = seed) 
  cv_ga_split = cv_ga.split(data_train, labels_train)

  fold_pred = [predictions[test] for train, test in cv_ga.split(data_train, labels_train)]
  fold_labels = [np.array(labels_train)[test] for train, test in cv_ga.split(data_train, labels_train)]
  data_train_sex = np.array(data_train['participants_sex'])
  i = 0
  fold_results = []
  for train_index, test_index in cv_ga_split:

    male_accuracy = 0
    male_total = 0
    female_accuracy = 0
    female_total = 0

    train = data_train_sex[train_index]
    test = data_train_sex[test_index]

    for index in range(len(fold_pred[i])): 
      if test[index] == 'M':
        male_total += 1
        if round(fold_pred[i][index]) == fold_labels[i][index]:
          male_accuracy += 1
      else:
        female_total += 1
        if round(fold_pred[i][index]) == fold_labels[i][index]:
          female_accuracy += 1
    i += 1
    male_ga = round(male_accuracy/male_total*100, 2)
    female_ga = round(female_accuracy/female_total*100, 2)
    ga_score = male_ga-female_ga
    print("Male: ", male_accuracy, " out of ", male_total,", ", male_ga, "%. Female: ", female_accuracy, " out of ", female_total, ", ", female_ga, "%. Total participants: ", female_total + male_total, sep="")
    if ga_score != 0:
      fold_results.append((np.abs(round(ga_score, 2)), round(abs(ga_score)/-ga_score))) #1 = F, -1 = M
    else:
      fold_results.append((np.abs(round(ga_score, 2)), 0)) #1 = F, -1 = M
  # print("Fold Results: ", fold_results)
  return fold_results

def train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor):
  #Create crossvalidation code
  folds = 1
  fold_results = []
  cv_custom = StratifiedKFold(n_splits=5, shuffle = True, random_state=42) 

  for train, test in cv_custom.split(data_train, labels_train):


    dataframe_indices = data_train.index.values.copy()
    train_dataset = pd.DataFrame(index = dataframe_indices[train]) #initialise dataframe with key values with unique acquisition

    train_dataset.loc[:, data_train.columns] = data_train.loc[dataframe_indices[train]] #Copy all dataframe information w.r.t key values
    train_dataset.index.name = "subject_id"

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
    folds += 1
  return fold_results


def general_accuracy(training_results, labels_test, sex_test):
  fold_results = {}
  i = 1
  for predicted_labels in training_results:
    ga_results = {}
    ga_results["overall"] = accuracy_score(labels_test, predicted_labels)

    male_results = predicted_labels[sex_test[0]]
    ga_results["male"] = accuracy_score(labels_test[sex_test[0]], male_results)

    female_results = predicted_labels[sex_test[1]]
    ga_results["female"] = accuracy_score(labels_test[sex_test[1]], female_results)
    
    fold_results[i] = ga_results.copy()
    i += 1


  return(fold_results)

def true_positive_rate(labels_test, predicted_labels):
  true_positive = 0
  false_negative = 0
  false_positive = 0
  true_negative = 0
  i = 0

  while i < len(labels_test):
    if labels_test[i] == 1 and predicted_labels[i] == 1:
      true_positive += 1
    elif predicted_labels[i] == 0 and labels_test[i] == 1:
      false_negative += 1
    elif predicted_labels[i] == 1 and labels_test[i] == 0:
      false_positive += 1
    elif labels_test[i] == 0 and predicted_labels[i] == 0:
      true_negative += 1
    i += 1
  print("TP:", true_positive, "FP:", false_positive, "FN:", false_negative, "TN:", true_negative, "Total:", true_positive+false_positive+false_negative+true_negative)
  if (true_positive + false_negative) != 0:
    return true_positive/(true_positive + false_negative)
  else:
    return 0
    

def equal_opportunity(training_results, labels_test, sex_test):
  fold_results = {}
  i = 1
  for predicted_labels in training_results:
    eo_results = {}
    # ga_results["overall"] = true_positive_rate(labels_test, predicted_labels)

    male_results = predicted_labels[sex_test[0]]
    female_results = predicted_labels[sex_test[1]]
    eo_results["overall"] = true_positive_rate(labels_test[sex_test[0]], male_results) - true_positive_rate(labels_test[sex_test[1]], female_results)
    
    fold_results[i] = eo_results.copy()
    i += 1

  return(fold_results)

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

def create_violin_graph(submission_results):
  sns.set(style="darkgrid", rc={'figure.figsize':(90, 10)})
  stuff = []
  index = submission_results.index.values.tolist()
  i = 0
  while i < len(submission_results.drop("Average", axis = 1).index):
    stuff.append(submission_results.drop("Average", axis = 1).iloc[i].to_numpy().transpose())
    print(i, index[i], stuff[i])
    i+=1
  results = pd.DataFrame(columns = ['test', 'results'])
  i = 0
  while i < len(index):
    j = 0
    while j < len(stuff[i]):
      container = pd.DataFrame({
        'test': [index[i]],
        'results': [stuff[i][j]]
        })
      results = pd.concat([results, container.copy()])
      j+=1
    i+=1

  
  # print(len(index))
  # print(len(stuff))
  print(results)
  # plot
  # print(submission_results.drop("Average", axis = 1))
  # print(stuff)
  plot = sns.violinplot(data = results, x = results['test'], y=results['results'])
# keys = group_results["Average"].index
# values = group_results["Average"]
  plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)
  plt.show()

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
  raw_submission_results = {}
  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  raw_submission_results["ga"] = general_accuracy(fold_results, labels_test, sex_test)
  raw_submission_results["eo"] = equal_opportunity(fold_results, labels_test, sex_test)
  print("Results:", raw_submission_results)

  organised_submission_results = organise_results(name, raw_submission_results)

  return organised_submission_results

def run_abethe_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "abethe_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.abethe_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.abethe_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  raw_submission_results = {}
  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  raw_submission_results["ga"] = general_accuracy(fold_results, labels_test, sex_test)
  raw_submission_results["eo"] = equal_opportunity(fold_results, labels_test, sex_test)
  print("Results:", raw_submission_results)

  organised_submission_results = organise_results(name, raw_submission_results)

  return organised_submission_results

def run_amicie_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "amicie_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.amicie_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.amicie_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  raw_submission_results = {}
  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  raw_submission_results["ga"] = general_accuracy(fold_results, labels_test, sex_test)
  raw_submission_results["eo"] = equal_opportunity(fold_results, labels_test, sex_test)
  print("Results:", raw_submission_results)

  organised_submission_results = organise_results(name, raw_submission_results)

  return organised_submission_results

def run_ayoub_ghriss_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "ayoub_ghriss_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.ayoub_ghriss_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.ayoub_ghriss_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  raw_submission_results = {}
  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  raw_submission_results["ga"] = general_accuracy(fold_results, labels_test, sex_test)
  raw_submission_results["eo"] = equal_opportunity(fold_results, labels_test, sex_test)
  print("Results:", raw_submission_results)

  organised_submission_results = organise_results(name, raw_submission_results)

  return organised_submission_results

def run_lbg_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "lbg_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.lbg_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.lbg_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  raw_submission_results = {}
  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  raw_submission_results["ga"] = general_accuracy(fold_results, labels_test, sex_test)
  raw_submission_results["eo"] = equal_opportunity(fold_results, labels_test, sex_test)
  print("Results:", raw_submission_results)

  organised_submission_results = organise_results(name, raw_submission_results)

  return organised_submission_results

def run_mk_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "mk_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.mk_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.mk_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  raw_submission_results = {}
  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  raw_submission_results["ga"] = general_accuracy(fold_results, labels_test, sex_test)
  raw_submission_results["eo"] = equal_opportunity(fold_results, labels_test, sex_test)
  print("Results:", raw_submission_results)

  organised_submission_results = organise_results(name, raw_submission_results)

  return organised_submission_results

def run_nguigui_original(data_train, labels_train, data_test, labels_test, sex_test, seed):
  name = "nguigui_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.nguigui_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.nguigui_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)

  raw_submission_results = {}
  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  raw_submission_results["ga"] = general_accuracy(fold_results, labels_test, sex_test)
  raw_submission_results["eo"] = equal_opportunity(fold_results, labels_test, sex_test)
  print("Results:", raw_submission_results)

  organised_submission_results = organise_results(name, raw_submission_results)

  return organised_submission_results

def run_Slasnista_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "Slasnista_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.Slasnista_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.Slasnista_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  raw_submission_results = {}
  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  raw_submission_results["ga"] = general_accuracy(fold_results, labels_test, sex_test)
  raw_submission_results["eo"] = equal_opportunity(fold_results, labels_test, sex_test)
  print("Results:", raw_submission_results)

  organised_submission_results = organise_results(name, raw_submission_results)

  return organised_submission_results

def run_vzantedeschi_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "vzantedeschi_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.vzantedeschi_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.vzantedeschi_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  raw_submission_results = {}
  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  raw_submission_results["ga"] = general_accuracy(fold_results, labels_test, sex_test)
  raw_submission_results["eo"] = equal_opportunity(fold_results, labels_test, sex_test)
  print("Results:", raw_submission_results)

  organised_submission_results = organise_results(name, raw_submission_results)

  return organised_submission_results

def run_wwwwmmmm_original(data_train, labels_train, data_test, labels_test, sex_test):
  name = "wwwwmmmm_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.wwwwmmmm_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.wwwwmmmm_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  raw_submission_results = {}
  fold_results = train_folds(data_train, labels_train, data_test, labels_test, Classifier, FeatureExtractor)
  raw_submission_results["ga"] = general_accuracy(fold_results, labels_test, sex_test)
  raw_submission_results["eo"] = equal_opportunity(fold_results, labels_test, sex_test)
  print("Results:", raw_submission_results)

  organised_submission_results = organise_results(name, raw_submission_results)

  return organised_submission_results


def run_tests(seed):
  np.random.seed(seed)
  #Load data
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  data_train, labels_train, data_test, labels_test = load_data()

  #Display initial dataset information
  # print(data_train.info())
  # print(labels_train.size)
  # print(data_test.info())
  # print(labels_test.size)

  #Display information of initial datasets with respect to acquisition sites.
  # print(data_train["participants_site"].value_counts().sort_index())
  # print(data_test["participants_site"].value_counts().sort_index())

  #Merge both intial datasets to preserve all datasets
  merged_dataset, merged_labels = join_original_datasets(data_train, labels_train, data_test, labels_test)

  #Display information regarding merged dataset
  # print(merged_dataset.info())
  # print(merged_labels.size)
  # print(merged_dataset["participants_site"].value_counts().sort_index()) #34
  # print(merged_dataset.index)

  #Generate randomised test dataset and remove from training dataset.
  new_train_dataset, new_train_labels, new_test_dataset, new_test_labels = separate_test_suite(merged_dataset, merged_labels)
  # print(new_test_dataset.head(5))

  sex_test = sex_index_split(new_test_dataset)
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
  
  if check_for_saved_file(seed) == False:
    #Train and test submissions

    # submissions = run_pearrr_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)
    # submissions = pd.concat([submissions, run_abethe_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    # submissions = pd.concat([submissions, run_amicie_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    # submissions = pd.concat([submissions, run_ayoub_ghriss_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    # submissions = pd.concat([submissions, run_lbg_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    # submissions = pd.concat([submissions, run_mk_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    submissions = run_nguigui_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test, seed)
    # submissions = pd.concat([submissions, run_nguigui_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    # submissions = pd.concat([submissions, run_Slasnista_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    # submissions = pd.concat([submissions, run_vzantedeschi_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
    # submissions = pd.concat([submissions, run_wwwwmmmm_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)])
  else:
    submissions = load_predictions(seed)

  print(submissions)

  create_violin_graph(submissions)
  fig= plt.figure(figsize=(90, 10))
  submissions["Average"].plot.bar()
  plt.xticks(rotation = 90)
  plt.legend(loc=(1.04, 0))
  plt.show()
  save_predictions(seed, submissions)

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

random_seed = 42
run_tests(random_seed)