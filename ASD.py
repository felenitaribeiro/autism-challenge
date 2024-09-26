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
import matplotlib.colors as mcolours
import matplotlib.lines as mlines

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
    return True
  return False
  

def save_predictions(seed, predictions):
  warnings.filterwarnings("ignore", message=".*`np.*` is a deprecated alias.*")
  predictions.to_csv("saved_outcomes/"+str(seed)+".txt", index = True)

def save_datasets(seed, new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test):
  warnings.filterwarnings("ignore", message=".*`np.*` is a deprecated alias.*")
  new_train_dataset.to_csv("datasets/" + str(seed) + "/train_dataset_" + str(seed) + ".txt", index = True)
  new_test_dataset.to_csv("datasets/" + str(seed) + "/test_dataset_" + str(seed) + ".txt", index = True)

  with open("datasets/" + str(seed) + "/train_labels_" + str(seed) + ".txt", 'w') as filehandle:
    filehandle.writelines(f"{train_labels}\n" for train_labels in new_train_labels)

  with open("datasets/" + str(seed) + "/test_labels_" + str(seed) + ".txt", 'w') as filehandle:
    filehandle.writelines(f"{test_labels}\n" for test_labels in new_test_labels)

  with open("datasets/" + str(seed) + "/sex_test_" + str(seed) + ".txt", 'w') as filehandle:
      filehandle.writelines(f"{sex}\n" for sex in sex_test) 


def load_predictions(seed):
  return pd.read_csv("saved_outcomes/"+str(seed)+".txt")

def load_datasets(seed):
  train_dataset = pd.read_csv("datasets/" + str(seed) + "/train_dataset_"+str(seed)+".txt")
  test_dataset = pd.read_csv("datasets/" + str(seed) + "/test_dataset_"+str(seed)+".txt")

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

def plot_violin(df, title, subtitle, split_string, seed, type):
  if type == "auc" or type == "ga":
    custom_colour_palette = ["#A14756", "#524C72"]
  else:
    custom_colour_palette = ["#007C5D", "#3A4C68"]


  sns.set_theme(style="whitegrid")
  sns.set_palette(custom_colour_palette)

  
  plot = sns.violinplot(data=df, x="Submissions", y="Results", split=True, hue=split_string, inner="stick")

  # # Function to add text labels for mean
  # def add_mean_labels(data, x_positions, ax):
  #     i = 0
  #     for x in x_positions:
  #         sub_data = data[data['Submissions'] == x]
  #         mean = sub_data['Results'].mean()

  #         text_x = i
  #         text_y = mean + 5
  #         text = f'Mean: {mean:.2f}'
  #         ax.text(text_x-0.25, text_y, text, ha='center', va='bottom', fontsize=10, color='black', rotation=90)
  #         i += 1

  # # Get the current Axes
  # ax = plt.gca()

  # # Get the x-positions for labels
  # x_positions = sorted(df['Submissions'].unique())

  # # Add mean labels
  # add_mean_labels(df, x_positions, ax)

  # Get the x-positions for labels
  # x_positions = sorted(df['Submissions'].unique())

  # # Add mean labels
  # add_mean_labels(df, x_positions, ax)
  plot.set_title(title +' Performance: Seed '+str(seed))
  plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)  
  plot.set_xlabel('Submissions')
  plot.set_ylabel(subtitle)
  plot.legend(loc=(1.040, 0.5))
  if split_string == "TPR:FPR":
    legend = plot.get_legend()
    new_labels = ['TPR$_m$ - TPR$_f$', 'FPR$_m$ - FPR$_f$'] 
    for text in legend.texts:
        text.set_text(new_labels[legend.texts.index(text)])
        
  

  plt.savefig('results/Seeded Results/'+str(seed)+'/'+title+'_violin.png', bbox_inches="tight")
  plt.draw()
  plt.show()

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
    # if split_names[1] != "overall":
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
  
  plot_violin(auc_df[auc_df["Sex"]!="overall"], "AUC-ROC", "AUC Performance (%)", "Sex", seed, "auc")
  create_ridgeline(auc_df[auc_df["Sex"]=="overall"], "AUC-ROC", "AUC Performance (%)", seed)

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
    # if split_names[1] != "overall":
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

  plot_violin(ga_df[ga_df["Sex"]!="overall"], "GA", "GA Performance (%)", "Sex", seed, "ga")
  create_ridgeline(ga_df[ga_df["Sex"]=="overall"], "GA", "GA Performance (%)",  seed)

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
      #male - female
      if test_name not in consolidated_test_names_tpr: #male
        consolidated_test_names_tpr.append(test_name)
        eo_tpr.append(tpr[l])
        eo_tpr_fpr.append("tpr")
      else: #female
        eo_tpr[consolidated_test_names_tpr.index(test_name)] -= tpr[l]
    l += 1
  # print(len(eo_tpr))
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
      #male - female
      if test_name not in consolidated_test_names_fpr: #male
        consolidated_test_names_fpr.append(test_name)
        eo_fpr.append(fpr[l])
        eo_tpr_fpr.append("fpr")
      else: #female
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


  eo_df = pd.DataFrame({"Submissions": generalised_submission_names,
                        "Results": np.concatenate([eo_tpr, eo_fpr]), 
                        "TPR:FPR": eo_tpr_fpr})

  return eo_df

def plot_eo_df(eo_df, seed):
  plot_violin(eo_df, "EO", "Difference in EO Performance [Male - Female] (%)", "TPR:FPR", seed, "eo")
  create_ridgeline(eo_df[(eo_df["TPR:FPR"]=="tpr")], "EO TPR", "Difference in TPR Performance [Male - Female] (%)", seed)
  create_ridgeline(eo_df[(eo_df["TPR:FPR"]=="fpr")], "EO FPR", "Difference in FPR Performance [Male - Female] (%)", seed)
  # create_ridgeline(eo_df, "EO", "Difference in EO Performance [Male - Female] (%)", seed)

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
  # print(name)
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
  # print(name)
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
  # print(name)
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
  # print(name)
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
  # print(name)
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
  # print(name)
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
  # print(name)
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
  # print(name)
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
  # print(name)
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
  # print(name)
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


def run_single_test(seed, new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test):
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

  submissions = []

 

  if check_for_saved_file(seed) == False:
    #Train and test submissions
    if seed == "42-basic_test_set":
      np.random.default_rng(seed=42)
    else:
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

  submissions["seed"] = seed
  auc_results = auc_roc(submissions)
  ga_results = general_accuracy(submissions)
  eo_results = equalised_odds(submissions)
  plot_auc_df(auc_results, seed)
  plot_ga_df(ga_results, seed)
  plot_eo_df(eo_results, seed)

  auc_results['seed'] = seed
  ga_results['seed'] = seed
  eo_results['seed'] = seed
  
  return auc_results, ga_results, eo_results
  


def run_multiple_test(seeds):
  seed_auc = []
  seed_ga = []
  seed_eo = []

  new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test = develop_test_set(42)
  tabularised_df(new_test_dataset, new_test_labels, 42)
  #collect all the test results for each auc, ga and eo.
  for seed in seeds:
    new_tests = run_single_test(seed, new_train_dataset, new_train_labels, new_test_dataset, new_test_labels, sex_test)
    seed_auc.append(new_tests[0])
    seed_ga.append(new_tests[1])
    seed_eo.append(new_tests[2])
  # print(seed_auc[0])
  i = 0
  merged_seed_auc = seed_auc[i]
  merged_seed_ga = seed_ga[i]
  merged_seed_eo = seed_eo[i]
  while i < len(seeds)-1:
    merged_seed_auc = pd.concat([merged_seed_auc, seed_auc[i+1]])
    merged_seed_ga = pd.concat([merged_seed_ga, seed_ga[i+1]])
    merged_seed_eo = pd.concat([merged_seed_eo, seed_eo[i+1]])
    i+=1

  merged_seed_auc['Seeded_Submissions'] = merged_seed_auc['Submissions'].astype(str).str.cat(merged_seed_auc['seed'].astype(str), sep='-')
  merged_seed_ga['Seeded_Submissions'] = merged_seed_ga['Submissions'].astype(str).str.cat(merged_seed_ga['seed'].astype(str), sep='-')
  merged_seed_eo['Seeded_Submissions'] = merged_seed_eo['Submissions'].astype(str).str.cat(merged_seed_eo['seed'].astype(str), sep='-')
  print(merged_seed_auc)
  i = 0
  for submission in merged_seed_auc["Submissions"].unique():

    if i == 0:
      submisssion_text = str(submission)
      df2print = merged_seed_auc[(merged_seed_auc["Submissions"]==submission)]
      i += 1
    else:
      submisssion_text = str(submisssion_text) + " & " + str(submission)
      df2print = pd.concat([df2print, merged_seed_auc[merged_seed_auc["Submissions"]==submission]])
      create_ridgeline_multiseed(df2print[df2print["Sex"]=="overall"], "AUC- "+ submisssion_text, "AUC Performance (%)", seeds)

      i=0
    print(submission)

  i = 0
  for submission in merged_seed_ga["Submissions"].unique():
    if i == 0:
      submisssion_text = str(submission)
      df2print = merged_seed_ga[(merged_seed_ga["Submissions"]==submission)]
      i += 1
    else:
      submisssion_text = str(submisssion_text) + " & " + str(submission)
      df2print = pd.concat([df2print, merged_seed_ga[merged_seed_ga["Submissions"]==submission]])
      create_ridgeline_multiseed(df2print[df2print["Sex"]=="overall"], "GA- "+ submisssion_text, "GA Performance (%)", seeds)

      i=0

  i = 0
  for submission in merged_seed_eo["Submissions"].unique():
    if i == 0:
      submisssion_text = str(submission)
      df2print = merged_seed_eo[(merged_seed_eo["Submissions"]==submission)]
      i += 1
    else:
      submisssion_text = str(submisssion_text) + " & " + str(submission)
      df2print = pd.concat([df2print, merged_seed_eo[merged_seed_eo["Submissions"]==submission]])
      create_ridgeline_multiseed(df2print[(df2print["TPR:FPR"]=="tpr")], "EO TPR- "+ submisssion_text, "Difference in TPR Performance [Male - Female] (%)", seeds)
      create_ridgeline_multiseed(df2print[(df2print["TPR:FPR"]=="fpr")], "EO FPR- "+ submisssion_text, "Difference in FPR Performance [Male - Female] (%)", seeds)

      i=0

def get_palette(type):
  if type == "auc" or type == "ga":
    start_color = "#A14756"
    end_color = "#524C72"
  else:
    start_color = "#007C5D"
    end_color = "#00B78F"

  start_rgb = mcolours.hex2color(start_color)
  end_rgb = mcolours.hex2color(end_color)

  interpolated_values = np.linspace(0, 1, 10)

  interpolated_colours = [
      mcolours.to_hex(np.array(start_rgb) + t * (np.array(end_rgb) - np.array(start_rgb)))
      for t in interpolated_values
  ]
  return interpolated_colours

def create_ridgeline(df, title, subtitle, seed):

  submission_dict = {}
  submission_names = df["Submissions"].unique()
  # print(submission_names)
  i = 0
  for name in submission_names:
    submission_dict[i] = name
    i+=1
  # print(submission_dict)

  sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}) 
  
  # Compute quartile values
  q25, q50, q75 = np.percentile(df["Results"], [25, 50, 75])

  colour_theme = get_palette("auc")
  sns.set_palette(colour_theme, 10, color_codes=True)

  # in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
  g = sns.FacetGrid(df, row="Submissions", hue="Submissions", aspect=15, height=1, palette=colour_theme)
  
  # then we add the densities kdeplots for each month
  g.map(sns.kdeplot, "Results", bw_adjust=1, clip_on=False, fill=True, alpha=1, linewidth=1.5)
  
  # here we add a white line that represents the contour of each kdeplot
  g.map(sns.kdeplot, 'Results', bw_adjust=1, clip_on=False, color="w", linewidth=2)
  
  # here we add a horizontal line for each plot
  g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

  # # # Add a rug plot to show individual data points
  # g.map(sns.rugplot, "Results", color="black", height=0.25)

  # Define and use a simple function to label the plot in axes coordinates
  def label(x, color, label):
      ax = plt.gca()
      ax.text(0, .2, label, fontweight="bold", color=color,
              ha="left", va="center", transform=ax.transAxes)


  g.map(label, "Submissions")

  # Function to add mean lines and annotations
  def add_mean_lines_and_annotations(data, color, label):
      ax = plt.gca()
      mean = data['Results'].mean()
      
      ax.axvline(mean, color=color, linestyle='--', label='Mean', ymin=0, ymax=0.65)
      ax.annotate(f'Mean: {mean:.2f}', xy=(mean, 0), xytext=(10, 45), textcoords='offset points', color='grey', fontweight='bold')

  # Add mean lines and annotations to each ridgeline
  g.map_dataframe(add_mean_lines_and_annotations)

  g.figure.subplots_adjust(hspace=-.25)
  
  # Remove axes details that don't play well with overlap
  g.set_titles("")
  g.set(yticks=[], ylabel="")
  g.despine(bottom=True, left=True)
  plt.xlabel(subtitle, fontweight='bold', fontsize=15)
  g.fig.suptitle(title + ": Seed " + str(seed), ha='right', fontsize=20, fontweight='bold')
  
  try:
    os.mkdir("results/Seeded Results/"+str(seed))
  except OSError as error:
    print("Folder " + str(seed) + " exists already, punk")
  plt.savefig('results/Seeded Results/'+str(seed)+"/"+title+'_ridgeline.png', bbox_inches="tight")
  plt.show()

# Process the dataframes into multiseed ridgeline plots across seeds.
def create_ridgeline_multiseed(df, title, subtitle, seed):

  submission_dict = {}
  submission_names = df["Submissions"].unique()
  # print(submission_names)
  i = 0
  for name in submission_names:
    submission_dict[i] = name
    i+=1
  # print(submission_dict)

  sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}) 
  
  # Compute quartile values
  q25, q50, q75 = np.percentile(df["Results"], [25, 50, 75])

  colour_theme = get_palette("auc")
  sns.set_palette(colour_theme, 10, color_codes=True)

  # in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
  g = sns.FacetGrid(df, row="Seeded_Submissions", hue="Seeded_Submissions", aspect=15, height=1, palette=colour_theme)
  
  # then we add the densities kdeplots for each month
  g.map(sns.kdeplot, "Results", bw_adjust=1, clip_on=False, fill=True, alpha=1, linewidth=1.5)
  
  # here we add a white line that represents the contour of each kdeplot
  g.map(sns.kdeplot, 'Results', bw_adjust=1, clip_on=False, color="w", linewidth=2)
  
  # here we add a horizontal line for each plot
  g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

  # Add a rug plot to show individual data points
  # g.map(sns.rugplot, "Results", color="black", height=0.25)

  # Define and use a simple function to label the plot in axes coordinates
  def label(x, color, label):
      ax = plt.gca()
      ax.text(-.1, .2, label, fontweight="bold", color=color,
              ha="left", va="center", transform=ax.transAxes)


  g.map(label, "Seeded_Submissions")

  # Function to add mean lines and annotations
  def add_mean_lines_and_annotations(data, color, label):
      ax = plt.gca()
      mean = data['Results'].mean()
      
      ax.axvline(mean, color=color, linestyle='--', label='Mean', ymin=0, ymax=0.65)
      ax.annotate(f'Mean: {mean:.2f}', xy=(mean, 0), xytext=(10, 45), textcoords='offset points', color='grey', fontweight='bold')

  # Add mean lines and annotations to each ridgeline
  g.map_dataframe(add_mean_lines_and_annotations)

  g.figure.subplots_adjust(hspace=-.05)
  
  # Remove axes details that don't play well with overlap
  g.set_titles("")
  g.set(yticks=[], ylabel="")
  g.despine(bottom=True, left=True)
  plt.xlabel(subtitle, fontweight='bold', fontsize=15)
  g.fig.suptitle(title + ": Seed " + str(seed), ha='center', fontsize=20, fontweight='bold')
  print(title)
  try:
    os.mkdir("results/Seeded Results/"+str(seed))
  except OSError as error:
    print("Folder " + str(seed) + " exists already, punk")
  plt.savefig(os.path.join('results', 'Seeded Results', str(seed), title + '_ridgeline.png'), bbox_inches="tight")
  plt.show() 

# Run the submissions' algorithms testing against the challenge's provided test set.
def basic_test_set():
  data_train, labels_train, data_test, labels_test = load_data()
  sex_test = sex_index_split(data_test)
  run_single_test("42-basic_test_set", data_train, labels_train, data_test, labels_test, sex_test)
  tabularised_df(data_test, labels_test, 42)

# Process datasets into CSV form with respect to acuisition site, sex and neurodiversity.
def tabularised_df(df, labels, seed):

  df["label"] = labels

  simple = df[["participants_site", "participants_sex", "label"]]
  grouped = simple.groupby("participants_site")
  # Initialize an empty DataFrame to store the results
  result_df = pd.DataFrame(columns=["participants_site", "participants_sex", "label", "count"])
  
  # Iterate through each group and compute value counts
  for group_name, group_data in grouped:
      counts = group_data[["participants_sex", "label"]].value_counts().reset_index()
      counts.columns = ["participants_sex", "label", "count"]
      counts["participants_site"] = group_name
      result_df = pd.concat([result_df, counts], ignore_index=True)

  result_df.set_index("participants_site", inplace=True)
  # First, create a pivot table to group by 'site', 'sex', and 'label' and sum the 'count'
  pivot_df = result_df.pivot_table(index=['participants_site', 'participants_sex'], columns='label', values='count', aggfunc='sum', fill_value=0)

  # Reset the index and rename the columns
  pivot_df = pivot_df.reset_index()
  pivot_df.columns = ['participants_site', 'participants_sex', 'neurotypical', 'neurodiverse']

  # If needed, you can sort the DataFrame
  pivot_df = pivot_df.sort_values(['participants_site', 'participants_sex']).reset_index(drop=True)
  pivot_df.set_index(['participants_site', 'participants_sex'], inplace=True)
  pivot_df.to_csv("datasets/" + str(seed) + "/test_dataset_stats_" + str(seed) + ".csv", index = True)
  print(pivot_df)

### Run tests with randomised seed 42 with standard test database
### The downloading of data is necessary the first time you run load_data()
basic_test_set()

### Run tests with randomised seed 42 with custom, test database
###If the seed you choose is not one of the ones investigated within the study (11, 42, 121, 1331, 14641 and 161051)
###The algorithms will require training and this will take time. (8 hours +- your machine capabilities)
###The downloading of data is necessary the first time you run load_data()
# seeds = [42]
# run_multiple_test(seeds)

### Run tests with multiple randomised seeds, 11, 121, 1331, 14641 and 161051 with custom, test database
###If the seed you choose is not one of the ones investigated within the study (11, 42, 121, 1331, 14641 and 161051)
###The algorithms will require training and this will take time. (8 hours +- your machine capabilities)
###The downloading of data is necessary the first time you run load_data()
# seeds = [11, 11*11, 11*11*11, 11*11*11*11, 11*11*11*11*11]
# run_multiple_test(seeds)

