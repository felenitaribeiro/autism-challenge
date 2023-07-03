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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

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


def evaluation(X, y, Classifier, FeatureExtractor):
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  pipe = make_pipeline(FeatureExtractor(), Classifier())
  cv = get_cv(X, y)
  # cv = StratifiedKFold(n_splits=5, random_state=42)
  results = cross_validate(pipe, X, y, scoring=['roc_auc', 'accuracy'], cv=cv,
                            verbose=1, return_train_score=True,
                            n_jobs=2)
  

  return results

def evaluation_predict(X,y, Classifier, FeatureExtractor):
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  # Note: in the cross_validate function, they use StratifiedShuffleSplit which allows for resampling
  pipe = make_pipeline(FeatureExtractor(), Classifier())
  cv_custom = StratifiedKFold(n_splits=5, shuffle = True, random_state=42) 
  
  results = cross_val_predict(pipe, X, y, cv=cv_custom,
                            verbose=1, n_jobs=2, method='predict')
  auc_roc_score = roc_auc_score(y, results)
  print("AUC-ROC Score:", auc_roc_score)
  return results

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


def general_accuracy(predictions, data_train, labels_train):
  
  print("General Accurracy: True Positive and True Negative Instances")
  # print(predictions)
  # print(cv_split)
  # print(data_train)
  # print(labels_train)

  warnings.filterwarnings("ignore", message=".*`np.*` is a deprecated alias.*")
  cv_ga = StratifiedKFold(n_splits=5, shuffle = True, random_state=42) 
  cv_ga_split = cv_ga.split(data_train, labels_train)

  fold_pred = [predictions[test] for train, test in cv_ga.split(data_train,labels_train)]
  fold_labels = [np.array(labels_train)[test] for train, test in cv_ga.split(data_train,labels_train)]
  data_train_sex = np.array(data_train['participants_sex'])
  i = 0
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
    
    print("Male: ", male_accuracy, " out of ", male_total,", ", round(male_accuracy/male_total*100, 2), "%. Female: ", female_accuracy, " out of ", female_total, ", ", round(female_accuracy/female_total*100, 2), "%. Total participants: ", female_total + male_total, sep="")


def equal_opportunity(predictions, data_train, labels_train):
  cv_eo = StratifiedKFold(n_splits=5, shuffle = True, random_state=42) 
  cv_eo_split = cv_eo.split(data_train, labels_train)
  print("Equal Opportunity: True Positive Instances")
  # print(predictions)
  # print(cv_split)
  # print(data_train)
  # print(labels_train)
  
  warnings.filterwarnings("ignore", message=".*`np.*` is a deprecated alias.*")

  fold_pred = [predictions[test] for train, test in cv_eo.split(data_train,labels_train)]
  fold_labels = [np.array(labels_train)[test] for train, test in cv_eo.split(data_train,labels_train)]
  data_train_sex = np.array(data_train['participants_sex'])
  i = 0
  for train_index, test_index in cv_eo_split:

    male_accuracy = 0
    male_total = 0
    female_accuracy = 0
    female_total = 0

    train = data_train_sex[train_index]
    test = data_train_sex[test_index]

    for index in range(len(fold_pred[i])): 

      if test[index] == 'M' and fold_labels[i][index] == 1:
        male_total += 1
        if round(fold_pred[i][index]) == fold_labels[i][index]:
          male_accuracy += 1
      elif test[index] == 'F' and fold_labels[i][index] == 1:
        female_total += 1
        if round(fold_pred[i][index]) == fold_labels[i][index]:
          female_accuracy += 1
    i += 1
    
    print("Male: ", male_accuracy, " out of ", male_total,", ", round(male_accuracy/male_total*100, 2), "%. Female: ", female_accuracy, " out of ", female_total, ", ", round(female_accuracy/female_total*100, 2), "%. Total participants: ", female_total + male_total, sep="")

def general_evaluation(data_train, labels_train, Classifier, FeatureExtractor):
  results = evaluation(data_train, labels_train, Classifier, FeatureExtractor)

  print("Training score ROC-AUC: {:.3f} +- {:.3f}".format(
    np.mean(results['train_roc_auc']), np.std(results['train_roc_auc'])))
  print("Validation score ROC-AUC: {:.3f} +- {:.3f} \n".format(
    np.mean(results['test_roc_auc']), np.std(results['test_roc_auc'])))

  print("Training score accuracy: {:.3f} +- {:.3f}".format(
    np.mean(results['train_accuracy']), np.std(results['train_accuracy'])))
  print("Validation score accuracy: {:.3f} +- {:.3f}".format(
    np.mean(results['test_accuracy']), np.std(results['test_accuracy'])))

def run_pearrr_original(data_train, labels_train, data_test, labels_test):
  print("pearrr_original")
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.pearrr_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.pearrr_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, "pearrr_original")
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_abethe_original(data_train, labels_train, data_test, labels_test):
  print("abethe_original")
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.abethe_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.abethe_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, "abethe_original")
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_amicie_original(data_train, labels_train, data_test, labels_test):
  print("amicie_original")
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.amicie_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.amicie_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, "amicie_original")
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_ayoub_ghriss_original(data_train, labels_train, data_test, labels_test):
  print("ayoub_ghriss_original")
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.ayoub_ghriss_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.ayoub_ghriss_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, "ayoub_ghriss_original")
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_lbg_original(data_train, labels_train, data_test, labels_test):
  print("lbg_original")
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.lbg_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.lbg_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, "lbg_original")
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_mk_original(data_train, labels_train, data_test, labels_test):
  print("mk_original")
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.mk_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.mk_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, "mk_original")
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train,  Classifier, FeatureExtractor)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_nguigui_original(data_train, labels_train, data_test, labels_test):
  print("nguigui_original")
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.nguigui_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.nguigui_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, "nguigui_original")
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_Slasnista_original(data_train, labels_train, data_test, labels_test):
  print("Slasnista_original")
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.Slasnista_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.Slasnista_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, "Slasnista_original")
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_vzantedeschi_original(data_train, labels_train, data_test, labels_test):
  print("vzantedeschi_original")
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.vzantedeschi_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.vzantedeschi_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, "vzantedeschi_original")
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
  general_accuracy(predictions, data_test, labels_test)
  equal_opportunity(predictions, data_test, labels_test)

def run_wwwwmmmm_original(data_train, labels_train, data_test, labels_test):
  print("wwwwmmmm_original")
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.wwwwmmmm_original.classifier import Classifier
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  from submissions.wwwwmmmm_original.feature_extractor import FeatureExtractor

  download_data()

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  predictions = initialise_predictions(data_train, labels_train, Classifier, FeatureExtractor)
  save_predictions(predictions, data_train, labels_train, "wwwwmmmm_original")
  # predictions = load_predictions("mk_original")

  # general_evaluation(data_train, labels_train, Classifier, FeatureExtractor)
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
  
  #New Dataframe Transfer of training data
  train_dataset = pd.DataFrame(index = subject_id_train[train_indices]) #initialise dataframe with key values with unique acquisition
  train_dataset.loc[:, overall_set.columns] = overall_set.loc[subject_id_train[train_indices]] #Copy all dataframe information w.r.t key values
  
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


warnings.filterwarnings("ignore", category=DeprecationWarning)
data_train, labels_train, data_test, labels_test = load_data()
# print(data_train.info())
# print(labels_train.size)
# print(data_test.info())
# print(labels_test.size)

# print(data_train["participants_site"].value_counts().sort_index())
# print(data_test["participants_site"].value_counts().sort_index())


merged_dataset, merged_labels = join_original_datasets(data_train, labels_train, data_test, labels_test)
# print(merged_dataset.info())
# print(merged_labels.size)
# merged_dataset["participants_site"].value_counts().sort_index() #34


new_train_dataset, new_train_labels, new_test_dataset, new_test_labels = separate_test_suite(merged_dataset, merged_labels)

# print(new_train_dataset.info())
# print(new_train_labels.size)
# print(new_test_dataset.info())
# print(new_test_labels.size)


# print(labels_train)
# print(labels_test)
# print(merged_labels)


# print(merged_dataset.loc[10631804530197433027])




# print_gender_info()
# gender_ratio_per_fold()

run_pearrr_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
run_abethe_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
run_amicie_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
run_ayoub_ghriss_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
run_lbg_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
run_mk_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
run_nguigui_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
run_Slasnista_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
run_vzantedeschi_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
run_wwwwmmmm_original(new_train_dataset, new_train_labels, new_test_dataset, new_test_labels)
