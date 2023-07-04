import warnings
import numpy as np
warnings.filterwarnings("ignore", message="Creating an ndarray from ragged nested sequences")
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

import matplotlib.pyplot as plt

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


# def evaluation(X, y, Classifier, FeatureExtractor):
#   warnings.filterwarnings("ignore", category=DeprecationWarning)
#   pipe = make_pipeline(FeatureExtractor(), Classifier())
#   cv = get_cv(X, y)
#   # cv = StratifiedKFold(n_splits=5, random_state=42)
#   results = cross_validate(pipe, X, y, scoring=['roc_auc', 'accuracy'], cv=cv,
#                             verbose=1, return_train_score=True,
#                             n_jobs=2)
  

#   return results

def evaluation_predict(X,y, Classifier, FeatureExtractor):
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  # Note: in the cross_validate function, they use StratifiedShuffleSplit which allows for resampling
  pipe = make_pipeline(FeatureExtractor(), Classifier())
  cv_custom = StratifiedKFold(n_splits=5, shuffle = True, random_state=42) 

  return cross_val_predict(pipe, X, y, cv=cv_custom, verbose=1, n_jobs=2, method='predict')

def gender_ratio_per_fold():
  #gender ratio per cross-validation fold
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
  data_train_sex = np.array(data_train['participants_sex'])
  fold_number = 1
  for train_index, test_index in cv.split(data_train,labels_train):
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

def save_predictions(predictions,data_train, labels_train,  file_name):
  warnings.filterwarnings("ignore", message=".*`np.*` is a deprecated alias.*")
  cv_save = StratifiedKFold(n_splits=5, shuffle = True, random_state=42) 
  cv_save_split = cv_save.split(data_train, labels_train)

  fold_pred = [predictions[test] for train, test in cv_save.split(data_train,labels_train)]
  fold_labels = [np.array(labels_train)[test] for train, test in cv_save.split(data_train,labels_train)]
  data_train_sex = np.array(data_train['participants_sex'])

  i = 0
  f = open("saved_outcomes/"+file_name+".txt", "w")
  for train_index, test_index in cv_save_split:

    test = data_train_sex[test_index]
    for index in range(len(fold_pred[i])): 
      f.write(str(fold_pred[i][index]))
      f.write(",")
      f.write(str(fold_labels[i][index]))
      f.write(",")
      f.write(test[index])
      f.write(",")
      f.write("\n")
    i += 1
  f.close()

def load_predictions(file_name):
  prediction_file = open("saved_outcomes/"+file_name+".txt", "r")
  
  predictions = []

  for line in prediction_file:
    predictions.append(line.split(","))

  return predictions

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

def general_accuracy(predictions, data_test, labels_test):
  
  print("General Accurracy: True Positive and True Negative Accuracy")
  # print(predictions)
  # print(cv_split)
  # print(data_test)
  # print(labels_test)

  warnings.filterwarnings("ignore", message=".*`np.*` is a deprecated alias.*")
  cv_ga = StratifiedKFold(n_splits=5, shuffle = True, random_state=42) 
  cv_ga_split = cv_ga.split(data_test, labels_test)

  fold_pred = [predictions[test] for train, test in cv_ga.split(data_test,labels_test)]
  fold_labels = [np.array(labels_test)[test] for train, test in cv_ga.split(data_test,labels_test)]
  data_test_sex = np.array(data_test['participants_sex'])
  i = 0
  fold_results = []
  for train_index, test_index in cv_ga_split:

    male_accuracy = 0
    male_total = 0
    female_accuracy = 0
    female_total = 0

    train = data_test_sex[train_index]
    test = data_test_sex[test_index]

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
    fold_results.append((np.abs(round(ga_score, 2)), round(abs(ga_score)/-ga_score))) #1 = F, -1 = M
  print("Fold Results: ", fold_results)
  return fold_results

def equal_opportunity(predictions, data_test, labels_test):
  cv_eo = StratifiedKFold(n_splits=5, shuffle = True, random_state=42) 
  cv_eo_split = cv_eo.split(data_test, labels_test)
  print("Equal Opportunity: Equal True Positive Rate")
  # print(predictions)
  # print(cv_split)
  # print(data_test)
  # print(labels_test)
  
  warnings.filterwarnings("ignore", message=".*`np.*` is a deprecated alias.*")

  fold_pred = [predictions[test] for train, test in cv_eo.split(data_test, labels_test)]  #predicted labels
  fold_labels = [np.array(labels_test)[test] for train, test in cv_eo.split(data_test, labels_test)]  #true labels
  data_test_sex = np.array(data_test['participants_sex'])
  i = 0
  fold_results = []
  for train_index, test_index in cv_eo_split:

    male_accuracy = 0
    male_total = 0
    female_accuracy = 0
    female_total = 0

    train_sex = data_test_sex[train_index]
    test_sex = data_test_sex[test_index]

    for index in range(len(fold_pred[i])): 

      if test_sex[index] == 'M' and round(fold_pred[i][index]) == 1:
        male_total += 1
        if round(fold_pred[i][index]) == fold_labels[i][index]:
          male_accuracy += 1
      elif test_sex[index] == 'F' and round(fold_pred[i][index]) == 1:
        female_total += 1
        if round(fold_pred[i][index]) == fold_labels[i][index]:
          female_accuracy += 1
    i += 1
    male_eo = round(male_accuracy/male_total*100, 2)
    female_eo = round(female_accuracy/female_total*100, 2)
    eo_score = male_eo-female_eo
    print("Male: ", male_accuracy, " out of ", male_total,", ", male_eo, "%. Female: ", female_accuracy, " out of ", female_total, ", ", female_eo, "%. Total (TP + FN) : ", female_total + male_total, sep="")
    fold_results.append((np.abs(round(eo_score, 2)), round(abs(eo_score)/-eo_score))) #1 = F, -1 = M
  print("Fold Results: ", fold_results)
  return fold_results



#Alternate, prior, method to determine aucroc score.
#No longer used due to keep cross-validation folds consistent.
# def general_evaluation(data_train, labels_train, Classifier, FeatureExtractor):
  # results = evaluation(data_train, labels_train, Classifier, FeatureExtractor)

  # print("Training score ROC-AUC: {:.3f} +- {:.3f}".format(
  #   np.mean(results['train_roc_auc']), np.std(results['train_roc_auc'])))
  # print("Validation score ROC-AUC: {:.3f} +- {:.3f} \n".format(
  #   np.mean(results['test_roc_auc']), np.std(results['test_roc_auc'])))

  # print("Training score accuracy: {:.3f} +- {:.3f}".format(
  #   np.mean(results['train_accuracy']), np.std(results['train_accuracy'])))
  # print("Validation score accuracy: {:.3f} +- {:.3f}".format(
  #   np.mean(results['test_accuracy']), np.std(results['test_accuracy'])))


#Functions to run each submission
def run_pearrr_original(data_train, labels_train, data_test, labels_test):
  name = "pearrr_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.pearrr_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.pearrr_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, name)
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  submission_results = []
  submission_results.append(plot_auc(labels_train, predictions, name))
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_abethe_original(data_train, labels_train, data_test, labels_test):
  name = "abethe_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.abethe_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.abethe_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, name)
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  plot_auc(labels_train, predictions, name)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_amicie_original(data_train, labels_train, data_test, labels_test):
  name = "amicie_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.amicie_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.amicie_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, name)
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  plot_auc(labels_train, predictions, name)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_ayoub_ghriss_original(data_train, labels_train, data_test, labels_test):
  name = "ayoub_ghriss_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.ayoub_ghriss_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.ayoub_ghriss_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, name)
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  plot_auc(labels_train, predictions, name)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_lbg_original(data_train, labels_train, data_test, labels_test):
  name = "lbg_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.lbg_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.lbg_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, name)
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  plot_auc(labels_train, predictions, name)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_mk_original(data_train, labels_train, data_test, labels_test):
  name = "mk_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.mk_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.mk_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, name)
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  plot_auc(labels_train, predictions, name)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_nguigui_original(data_train, labels_train, data_test, labels_test):
  name = "nguigui_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.nguigui_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.nguigui_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, name)
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  plot_auc(labels_train, predictions, name)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_Slasnista_original(data_train, labels_train, data_test, labels_test):
  name = "Slasnista_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.Slasnista_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.Slasnista_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, name)
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  plot_auc(labels_train, predictions, name)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_vzantedeschi_original(data_train, labels_train, data_test, labels_test):
  name = "vzantedeschi_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.vzantedeschi_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.vzantedeschi_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, name)
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  plot_auc(labels_train, predictions, name)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_wwwwmmmm_original(data_train, labels_train, data_test, labels_test):
  name = "wwwwmmmm_original"
  print(name)
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.wwwwmmmm_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.wwwwmmmm_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, name)
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  plot_auc(labels_train, predictions, name)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def separate_test_suite(overall_set, overall_labels):
  # print(overall_labels.size)
  #Gather all indices which are unique. (w.r.t site, sex and neurostatus)
  test_indices = determine_test_sample_indicies(overall_set, overall_labels)
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

  
def determine_test_sample_indicies(overall_set, overall_labels):
  np.random.seed(42)
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

#Train and test submissions
# run_pearrr_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
# run_abethe_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
# run_amicie_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
# run_ayoub_ghriss_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
# run_lbg_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
# run_mk_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
run_nguigui_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
# run_Slasnista_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
# run_vzantedeschi_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
# run_wwwwmmmm_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
