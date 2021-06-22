############################ IMPORTS ###########################################

import streamlit as st

import numpy as np
import pandas as pd
import re

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from scipy.spatial.distance import cdist

import random
import time
import datetime


########################## CALCULATE RATIO #####################################
def ratio(a, b):
  """
    Calculate smallest common multiple
    :param a (int): first number for ratio
    :param b (int): second number for ratio
    :return: a (int): smallest common multiple
    Source: https://gist.github.com/svschannak/8510014 (09.06.2021)
  """
  a = float(a)
  b = float(b)
  if b == 0:
      return a
  return ratio(b, a % b)

def get_ratio(a, b):
  """
    Calculate ratio of two numbers
    :param a (int): first number for ratio
    :param b (int): second number for ratio
    :return: Streamlit write (minority:majority, ratio)
    Source: https://gist.github.com/svschannak/8510014 (09.06.2021)
  """
  r = ratio(a, b)
  ar = int(a/r)
  br = int(b/r)
  
  distr = str(str(ar) + " : " + str(br))
  
  return st.write("Ratio Majority : Minority samples  \n", distr, "(Ratio", (round(ar/br,2)), ")")


########################## DISPLAY COUTPLOT #####################################

def display_countplot(data, target, col_index=None, cols=None):
  """
    Display countplot of certain column (mostly target column)
    :param data (dataframe): data for countplot
    :param target (str): label of target column
    :param col_index (int): index of column, if plot should be displayed in column
    :param cols (streamlit column object): column object, need col_index
    :return: display pyplot countplot
    Source: https://stackoverflow.com/questions/31749448/how-to-add-percentages-on-top-of-bars-in-seaborn (09.06.2021)
  """
  ax = sns.countplot(x=target, data=data)
  if target == 'term_deposit':
    ax.set_xlabel('Distribution of Target Variable: Subscription to Term Deposit')
  else:
    ax.set_xlabel('Distribution of Target Variable: Income >50K')
  #plt.figure(figsize=(2,2))
  fig = ax.get_figure()
  for p in ax.patches:
      percentage = (p.get_height())
      percentage = '{:.1f}%'.format(100 * p.get_height()/(len(data)))
      x = p.get_x() + p.get_width()
      y = p.get_height()
      ax.annotate(percentage, (x, y),ha='right', fontsize=15, xytext = (0, -15), textcoords = 'offset points', color="white")
  if col_index == None:
    st.pyplot(fig,clear_figure=True)
  else:
    cols[col_index].pyplot(fig, clear_figure=True)
  

def save_runtime_tofile(runtime, algorithm, data_selection):
  """
    Save runtime to .txt
    :param runtime (timediff): runtime to write to log file
    :param algorithm (str): label for used algorithm
    :param data_selection (str): label for used data
    :return: a (int): smallest common multiple
    Source: https://gist.github.com/svschannak/8510014 (09.06.2021)
  """
  # Source: https://stackoverflow.com/questions/12400256/converting-epoch-time-into-the-datetime
  timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
  with open('runtime_log.txt', "a") as txt_file:
      txt_file.write('| Timestamp: ' +  str(timestamp) + '\t' +
                     '| Data: ' +  data_selection + '\t' +
                     '| Algorithm: ' + algorithm + '\t' +
                     '| Runtime in s: ' + str(runtime) + '\n')


########################## Undersampling #####################################
  
def create_clusters(X, k=10, max_iter=300):
  """
    Create clusters using kmeans++
    :param X (dataframe): data to be clustered
    :param k (int): number of clusters
    :param max_iter (int): maximum iterations for k-means
    :return: clusters (KMeans cluster object)
    Source: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html (09.06.2021)
  """
  clusters = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=max_iter)
  clusters.fit(X)
  return clusters



def undersample(data_train, target, undersampling_algorithm, k, max_iter, size_minority):
  """
    Create clusters using kmeans++
    :param data_train (dataframe): data to be balanced
    :param target (str): label of target column
    :param undersampling_algorithm (str): label of undersamling algorithm to be applied
    :param k: number of clusters
    :param max_iter (int): maximum iterations for k-means++
    :size_minority (int): number of instances of the minority class in data_train
    :return: balanced_data_train (dataframe): majority and minority class both have size_minority samples
  """

  pd.options.mode.chained_assignment = None  # default='warn'

#    ("K-means++ and Centroids", 
#  "K-means++ and Random sampling",
#  "K-means++ and Top1 centroids' nearest neighbor",
#  "K-means++ and TopN centroids' nearest neighbors"))

  save = False
  # will be true for K-means++ and Random sampling and K-means++ and TopN centroids' nearest neighbors
  # goal: select indices of representatives to keep
  select_representatives = False
  representatives = []
  
# ---PREPARE DATA---
  majority = data_train[data_train[target] == 0]
  minority = data_train[data_train[target] == 1]
    
  majority_X = majority.loc[:, majority.columns != target]
  majority_y = majority.loc[:, majority.columns == target]
  
  majority["indices"] = majority.index # create column with indices in dataframe
  feature_columns = majority_X.columns # independent variable labels
  
#---CLUSTER------
  
  clusters = create_clusters(majority_X, int(k), max_iter)
  centroids = clusters.cluster_centers_
  majority["cluster"] = clusters.labels_

#---SET N (number of samples to take from each cluster)-----
  
  N = size_minority//k
  rest = size_minority - k*N # rest of N, if size_minority is not a multiple of k --> to be filled from clusters with len > N
  
#---DISTINGUISH PROCEDURE FOR UNDERSAMPLING ALGORITHMS-----

  if undersampling_algorithm == "K-means++ and Centroids": #-----------------------------
    # use centroids as representatives, prepare to save balanced data as csv
    balanced_majority = pd.DataFrame(data=centroids, columns=feature_columns)
    balanced_majority[target] = 0
    save = True
    save_name = "centroids"
  
  elif undersampling_algorithm == "K-means++ and Random sampling": #---------------------
    # create list with indices of samples belonging to each cluster
    cluster_idx_group = majority.groupby('cluster').agg({'indices':lambda x: list(x)}) # get indices of elements in clusters
    cluster_idx = cluster_idx_group.indices.to_list()
    
    for cluster in cluster_idx:
      random.shuffle(cluster) # shuffle indices belonging to each cluster
      
    select_representatives = True
      
  #---CALCULATE AND SORT DISTANCE TO CENTROIDS----
  else:
    # calculate euclidean distance of each sample to its cluster centroid
    majority["distance"] = None
    for x in range(k):
      samples_current_cluster = majority[majority.cluster==x].drop(columns=['cluster', 'indices', 'distance', target])
      majority["distance"][majority.cluster==x] = cdist([centroids[x]], (samples_current_cluster.values), metric = "euclidean")[0]
    
    # order majority dataframe by cluster and distance to centroids
    sorted_majority = majority.sort_values(by=['cluster', 'distance'])
    
  #-----------------
    
    if undersampling_algorithm == "K-means++ and Top1 centroids' nearest neighbor": #-----------------
      # only keep first sample (closest to centroid) for each cluster (=Top1 neighbor), prepare to save balanced data as csv
      balanced_majority = sorted_majority.drop_duplicates(['cluster'], keep="first").drop(columns=['cluster', 'indices', 'distance'])
      save = True
      save_name = "top1_centroidsNN"    
    
    else: # K-means++ and TopN centroids' nearest neighbors #-------------------------------------
      # create list with indices of samples belonging to each cluster (sorted by distance)
      cluster_idx_group = sorted_majority.groupby('cluster').agg({'indices':lambda x: list(x)}) # get indices of elements in clusters
      cluster_idx = cluster_idx_group.indices.to_list()
      
      select_representatives = True
   
     
#---SELECT REPRESENTATIVES FOR K-means++ and Random sampling AND K-means++ and TopN centroids' nearest neighbors----
  
  if select_representatives:
    #---take N examples out of each cluster--- (remember number of elements that could not be taken because the cluster was to small)
    for x in range(k): # once for all clusters
      if len(cluster_idx[x]) > N:
          representatives.extend(cluster_idx[x][0:N]) # add N elements from cluster to majority representatives
          del cluster_idx[x][0:N] # remove chosen elements
      else:
          rest = rest + (N - len(cluster_idx[x])) # add missing elements in cluster to 'rest' (difference to N)
          representatives.extend(cluster_idx[x][:]) # if number of instances in cluster is < N, then add all elements
          del cluster_idx[x][:] # remove chosen elements

# "rest" is the difference of representatives that I found so far to the number of minority class instance

    #---select "rest" iteratively from clusters with remaining instances--- (iteratively one element from each cluster)
    count = 0
    current_cluster = 0
    
    while count < rest:
        if len(cluster_idx[current_cluster]) > 0:
            idx = cluster_idx[current_cluster].pop(0)
            representatives.append(idx)
            count += 1       
        current_cluster +=1
        if current_cluster == k: 
            current_cluster = 0 # restart with first cluster (if we tried every cluster)
            
    #---extract chosen representatives from majority data----
    if undersampling_algorithm == "K-means++ and TopN centroids' nearest neighbors":
      balanced_majority = majority.loc[representatives, majority.columns].drop(columns=['cluster', 'indices', 'distance'])
    else: # K-means++ and Random sampling
      balanced_majority = majority.loc[representatives, majority.columns].drop(columns=['cluster', 'indices'])
    
#---CONCAT UNDERSAMPLED MAJORITY CLASS AND MINORITY CLASS SAMPLES---    
  balanced_data_train = pd.concat([balanced_majority, minority], ignore_index=True)

#---SAVED BALANCED TRAINING DATA--- 
  if save:
    target_savename = re.sub('[^A-Za-z0-9_]+', '', target)
    balanced_data_train.to_csv("data/" + target_savename + "_" + save_name + '_balanced_data_train.csv', index=False) 
  
  return balanced_data_train


########################## CLASSIFIER #####################################

def classifier(data_train, data_test, target):
  """
    Classify data using logistic regression
    :param data_train (dataframe): data to be classified
    :param data_test (dataframe): data for evaluation
    :param target (str): label of target column
    :return: 
      eval_metrics (dict): dictionary containing accuracy, precision, recall, f1-score and AUC
      matrix (confusion matrix): resulting confusion matrix
      fpr_roc (ndarray): increasing false positive rate with regard to threshold
      tpr_roc (ndarray): increasing true positive rate with regard to threshold
    Source: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8 (Last Access 11.06.2021)
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
  """

#---SPLIT FEATURES FROM TARGET-----

  X_train = data_train.loc[:, data_train.columns != target].values
  y_train = data_train.loc[:, data_train.columns == target].values.ravel()
  
  X_test = data_test.loc[:, data_test.columns != target].values
  y_test = data_test.loc[:, data_test.columns == target].values.ravel()

#---FIT LOGISTIC REGRESSION-----

  logreg = LogisticRegression(max_iter=300)
  logreg.fit(X_train, y_train)
  y_pred = logreg.predict(X_test)
  y_probs = logreg.predict_proba(X_test)
  
#---EVALUATION-----
#
  matrix = confusion_matrix(y_test, y_pred)

  eval_report = classification_report(y_test, y_pred, output_dict=True)
  
  accuracy = eval_report["accuracy"]
  precision = eval_report["macro avg"]["precision"]
  recall = eval_report["macro avg"]["recall"]
  f1 = eval_report["macro avg"]["f1-score"]
  auc = roc_auc_score(y_test, y_probs[:,1])
  
  eval_metrics = {"accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1, "auc":auc}
  
  
  fpr_roc, tpr_roc, thresholds = roc_curve(y_test, y_probs[:,1])

#--------

  return eval_metrics, matrix, fpr_roc, tpr_roc




##############################################################################

##################        START OF STREAMLIT APP       #######################

##############################################################################

#---SET INITIAL PARAMETERS-----

st.set_page_config(layout="centered")

test_size=0.2
max_iter=500

k = None
undersampling_algorithm = None
undersampling_submitted = False
recalculate = True

adult_data =  pd.read_csv('data/adult/adult_preprocessed.csv', sep=",")
bank_data =  pd.read_csv('data/bank/bank_preprocessed.csv', sep=",")


################DATA SELECTION#########################

with st.sidebar:
  st.title('Parameter Selection')
  
  mode = st.radio("Select App Mode:",
  ("Data Exploration", 
  "Classification Results",
  "Undersampling Evaluation"), index=1)

  data_selection = st.selectbox("Data Selection:", ("Adult Data", "Bank Data"))
  
  if data_selection == "Adult Data":
    orig_data = adult_data
    target = 'income_>50k'
    columns_to_scale = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    # to be able to change type of binary columns to object for statistics:
    type_dict = {'sex_female': object, 'income_>50k': object} # to be able to change type of binary columns to object for statistics}
  else:
    orig_data = bank_data
    target = 'term_deposit'
    columns_to_scale = ['age', 'balance', 'duration', 'campaign', 'previous']
    # to be able to change type of binary columns to object for statistics:
    type_dict = {"loan":object, "default":object, "housing":object, "term_deposit":object}
  
  data = pd.get_dummies(orig_data)  
  data[columns_to_scale] = data[columns_to_scale].apply(lambda x:(x-x.min()) / (x.max()-x.min()))

  whole_majority = len(data[data[target] == 0])
  whole_minority = len(data[data[target] == 1])
  
  get_ratio(whole_majority, whole_minority)
  
  
################MODE CLASSIFICATION RESULTS#########################
  
  if mode == "Classification Results":
    
#---SET PARAMS (test size, max_iter)------
    st.markdown("""---""")
    params_expander = st.beta_expander("Set Test Size and max. Iterations", expanded=False)
    with params_expander:
      with st.form("test_size"):
        test_size = st.slider("Test-Size (Fraction of Dataset:)", 0.0, 1.0, 0.2)
        max_iter = st.slider("Maximum Iterations for k-means++", 0, 1000, 500)
        test_size_submitted = st.form_submit_button("Submit")

    st.write("--> Current Test Size:", test_size)
    st.write("--> Current Maximum Iterations:", max_iter)
    st.markdown("""---""")

#### SPLIT DATASET ###
  
data_train, data_test = train_test_split(data, test_size=test_size, random_state=1234)

size_minority_train = len(data_train[data_train[target] == 1])
size_majority_train = len(data_train[data_train[target] == 0])

#####################

if mode == "Classification Results":
  st.title('Balancing of a Skewed Dataset')
  #st.write(size_majority_train)
  #st.write(size_minority_train)
#---CHOOSE UNDERSAMLING PARAMETERS-----
# (k, Algorithm)

  with st.sidebar:
    st.markdown("""**Set Parameters for Undersampling:**""")
    
    set_undersampling_algorithm = st.radio("Select Undersampling Method:",
    ("K-means++ and Centroids", 
    "K-means++ and Top1 centroids' nearest neighbor",
    "K-means++ and Random sampling",
    "K-means++ and TopN centroids' nearest neighbors"),index=0)
    
    if set_undersampling_algorithm in ["K-means++ and Random sampling", "K-means++ and TopN centroids' nearest neighbors"]:
      set_k = st.number_input('Set k:', 0, size_minority_train, 10)
    else:
      set_k = size_minority_train
      st.write("k =", size_minority_train, ("(size of minority class in training data)"))
      
      # only provide option to recalculate if params match presaved balanced dataset
      # (otherwise always recaltulate)
      if (test_size == 0.2) & (max_iter ==500): 
        recalculate = st.checkbox("Recalculate")
    
    if st.button("Submit"):
      k = set_k
      undersampling_algorithm = set_undersampling_algorithm
      undersampling_submitted = True
      
#---DISPLAY CURRENT CONFIGURATION----
  
  st.markdown("""---""")
  st.write("--> Current Undersampling Algorithm:", undersampling_algorithm)
  st.write("--> Current k for kmeans++:", k)
  st.markdown("""---""")
  

#---DISPLAY INITIAL TRAINING TARGET DISTRIBUTION---- 

  cols = st.beta_columns(2) 
  
  cols[0].header("Initial distribution (training data):")
  display_countplot(data_train, target, 0, cols)

#---APPLY UNDERSAMPLING WITH CHOSEN CONFIG---- 

  if undersampling_submitted:
    if recalculate:
      start = time.time()
      #start = datetime.datetime.now()
      
      balanced_data_train = undersample(data_train, target, undersampling_algorithm, k, max_iter, size_minority_train)
    
      stop = time.time()
      timediff_undersample = stop - start
      
      save_runtime_tofile(timediff_undersample, undersampling_algorithm, data_selection)
      
      st.write("Undersampling Runtime: " + str(round(timediff_undersample,2)) + "s")
      #st.write("Undersampling Runtime: " + str(timediff_undersample))
      
      
    else:
      target_savename = re.sub('[^A-Za-z0-9_]+', '', target)
      if undersampling_algorithm == "K-means++ and Centroids":
        balanced_data_train = pd.read_csv("data/" + target_savename + "_centroids_balanced_data_train.csv", sep=",")
      else:
        balanced_data_train = pd.read_csv("data/" + target_savename + "_top1_centroidsNN_balanced_data_train.csv", sep=",")
      
    cols[1].header("Balanced distribution (training data):")
    display_countplot(balanced_data_train, target, 1, cols)
    
#---EVALUATE PERFORMANCE BALANCED VS. UNBALANCED DATA------

    start = time.time()
    
    # unbalanced
    eval_metrics, conf_matrix, fpr_roc, tpr_roc = classifier(data_train, data_test, target)
    
    stop = time.time()
    timediff_unbalanced = stop - start
    
    save_runtime_tofile(timediff_unbalanced, "logreg_unbalanced", data_selection)
    
    #----
    
    start = time.time()
    
    # balanced
    eval_metrics_balanced, conf_matrix_balanced, fpr_roc_balanced, tpr_roc_balanced = classifier(balanced_data_train, data_test, target)

    stop = time.time()
    timediff_balanced = stop - start
    
    save_runtime_tofile(timediff_balanced, "logreg_balanced", data_selection)
    
    #---
    
    st.markdown("""---""")
    st.title("Performance Comparison - Unbalanced (Initial) Data vs. Balanced Data")
  
    cols = st.beta_columns(2)
    
    cols[0].write("Classification Runtime Unbalanced: " + str(round(timediff_unbalanced,2)) + "s")
    cols[0].write("Confusion Matrix Initial Data:")
    cols[0].write(conf_matrix)
    
    cols[1].write("Classification Runtime Balanced: " + str(round(timediff_balanced,2)) + "s")
    cols[1].write("Confusion Matrix Balanced Data:")
    cols[1].write(conf_matrix_balanced)
    
    st.markdown("""---""")

    cols = st.beta_columns(2) 
  
    # Source: https://towardsdatascience.com/mastering-the-bar-plot-in-python-4c987b459053 (11.06.2021)
    
    #---CREATE BARPLOT FOR PERFORMANCE COMPARISON BALANCED VS. UNBALANCED DATA-----
    results = list(eval_metrics.values())
    results_balanced = eval_metrics_balanced.values()
    metric_labels = list(eval_metrics.keys())
    
    bars = np.arange(len(results))
    
    fig_eval, ax_eval = plt.subplots()
    plt.bar(bars, results, color = 'C0', width = 0.25, label = "Initial Data")
    plt.bar(bars + 0.25, results_balanced, color = 'C1', width = 0.25, label = "Balanced Data")
    
    plt.xticks([i + 0.125 for i in range(len(results))], metric_labels)
    
    ax_eval.legend(loc='lower right')#, bbox_to_anchor=(0.5, 0.5))
    
    plt.title("Prediction Metrics Initial vs. Balanced Data")
    plt.xlabel('Metric')
    plt.ylabel('Result')
    
    for p in ax_eval.patches:
        percentage = '{:.2f}%'.format(100* p.get_height())
        x = p.get_x() + p.get_width()
        y = p.get_height()
        ax_eval.annotate(percentage, (x, y),ha='right', fontsize=10, rotation=90, xytext = (0, -37), textcoords = 'offset points', color="white")    
    
    cols[0].pyplot(fig_eval, clear_figure=True)
    
    #---CREATE ROC CURVE FOR PERFORMANCE COMPARISON BALANCED VS. UNBALANCED DATA-----
    # Source: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8 (Last Access 11.06.2021)
    
    fig_roc, ax_roc = plt.subplots()
    plt.plot(fpr_roc, tpr_roc, label='Initial Data (auc = {:.4f})'.format(eval_metrics["auc"]))
    plt.plot(fpr_roc_balanced, tpr_roc_balanced, label='Balanced Data (auc = {:.4f})'.format(eval_metrics_balanced["auc"]))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    ax_roc.legend(loc="lower right")
    
    cols[1].pyplot(fig_roc, clear_figure=True)
    
  st.markdown("""---""")
  
  
      


#################### MODE UNDERSAMPLING EVALUATION ############################


if mode == "Undersampling Evaluation":
  with st.sidebar:
    st.write("--> Test-Size:", test_size) # fixed test size (0.2)

    st.write("Initial Distribution (training data):")
    display_countplot(data_train, target)
    
    # display target distribution (size_minority = size_majority)
    st.write("Target Distribution after Undersampling (training data):")
    list_balanced = [0,1] * size_minority_train
    df_balanced = pd.DataFrame(list_balanced, columns=[target])
    display_countplot(df_balanced, target)
  

#---APPLY ALL UNDERSAMPLING ALGORITHMS AND RECORD PERFORMANCE-----
  
  st.title("Undersampling Evaluation")
  
  algorithms = ["K-means++ and Centroids", 
                "K-means++ and Top1 centroids' nearest neighbor", 
                "K-means++ and Random sampling", 
                "K-means++ and TopN centroids' nearest neighbors"]
  

  #--- Parameter Selection-----
  
  with st.form("eval_params"):
    st.write("Size of Minority Class in Training Data:", size_minority_train)
    st.markdown("""---""")
    st.write("1.", algorithms[0] + ": k = ", size_minority_train)
    st.write("2.", algorithms[1] + ": k = ", size_minority_train)
    recalculate_eval = st.checkbox("Recalculate")
    st.markdown("""---""")    
    
    k_random_sampling = st.number_input(("3. " + algorithms[2] + ' --> Set k:'), 0, size_minority_train, 10, key="k_random_sampling")
  
    k_topN_centroids = st.number_input(("4. " + algorithms[3] + ' --> Set k:'), 0, size_minority_train, 10, key="k_topN_centroids")
    eval_params_submitted = st.form_submit_button("Submit")

  #---EVALUATE UNBALANCED PERFORMANCE-----
  
  if eval_params_submitted:
    results_dict = {}
    
    eval_metrics, conf_matrix, fpr_roc, tpr_roc = classifier(data_train, data_test, target)
    results_dict["Unbalanced"] = eval_metrics 
    
  #---OBTAIN BALANCED DATA FOR "K-means++ and Centroids" AND "K-means++ and Top1 centroids' nearest neighbor"-----
  
    for undersampling_algorithm in algorithms[0:2]:
      k = size_minority_train
      
      # recalculate or...
      if recalculate_eval:
        k = size_minority_train
        balanced_data_train = undersample(data_train, target, undersampling_algorithm, k, max_iter, size_minority_train)
      
      # ... load saved balanced dataset 
      else:
        target_savename = re.sub('[^A-Za-z0-9_]+', '', target)
        if undersampling_algorithm == "K-means++ and Centroids":
          balanced_data_train = pd.read_csv("data/" + target_savename + "_centroids_balanced_data_train.csv", sep=",")
        else:
          balanced_data_train = pd.read_csv("data/" + target_savename + "_top1_centroidsNN_balanced_data_train.csv", sep=",")
      
      #---OBTAIN AND ADD PERFORMANCE RESULTS TO results_dict
      eval_metrics, conf_matrix, fpr_roc, tpr_roc = classifier(balanced_data_train, data_test, target)
      results_dict[undersampling_algorithm] = eval_metrics 
      
  #---OBTAIN BALANCED DATA AND PERFORMANCE METRICS FOR "K-means++ and Random sampling" 
  
    balanced_data_train = undersample(data_train, target, algorithms[2], k_random_sampling, max_iter, size_minority_train)
    eval_metrics, conf_matrix, fpr_roc, tpr_roc = classifier(balanced_data_train, data_test, target)
    results_dict[algorithms[2]] = eval_metrics
    
  #---OBTAIN BALANCED DATA AND PERFORMANCE METRICS FOR "K-means++ and TopN centroids' nearest neighbors"
  
    balanced_data_train = undersample(data_train, target, algorithms[3], k_topN_centroids, max_iter, size_minority_train)
    eval_metrics, conf_matrix, fpr_roc, tpr_roc = classifier(balanced_data_train, data_test, target)
    results_dict[algorithms[3]] = eval_metrics
    
    
#---PLOT RESULTS-----
# Source: https://towardsdatascience.com/mastering-the-bar-plot-in-python-4c987b459053 (11.06.2021)
    
  #---CREATE BARPLOT bars grouped by evaluation metric-----    
    groupby_metric_expander = st.beta_expander("Comparison Results Grouped by Evaluation Metric", expanded=True)
    with groupby_metric_expander:
      plotted_algos = ["Unbalanced",
                       "K-means++ and\n Centroids", 
                       "K-means++ and\n Top1 centroids' \nnearest neighbor", 
                       "K-means++ and \nRandom sampling", 
                       "K-means++ and\n TopN centroids' \nnearest neighbors"]
      
      unbalanced = []
      unbalanced.extend(list(results_dict["Unbalanced"].values()))
        
      centroids = []
      centroids.extend(list(results_dict["K-means++ and Centroids"].values()))
  
      top1 = []
      top1.extend(list(results_dict["K-means++ and Top1 centroids' nearest neighbor"].values()))
      
      random = []  
      random.extend(list(results_dict["K-means++ and Random sampling"].values()))
      
      topn = []
      topn.extend(list(results_dict["K-means++ and TopN centroids' nearest neighbors"].values()))
      
      bars = np.arange(len(unbalanced))
      
      fig_eval, ax_eval = plt.subplots()
      plt.bar(bars, unbalanced, color = 'C0', width = 0.15, label = "Unbalanced")
      plt.bar(bars + 0.17, centroids, color = 'C1', width = 0.15, label = "K-means++ & Centroids")
      plt.bar(bars + 0.34, top1, color = 'C2', width = 0.15, label = "K-means++ & Top1 centroids' nearest neighbor")
      plt.bar(bars + 0.51, random, color = 'C3', width = 0.15, label = "K-means++ & Random sampling")
      plt.bar(bars + 0.68, topn, color = 'C4', width = 0.15, label = "K-means++ & TopN centroids' nearest neighbors")
      
      plt.xticks([i + 0.35 for i in range(len(results_dict["Unbalanced"]))], list(results_dict["Unbalanced"].keys()), rotation=0, fontsize=7)
      
      ax_eval.legend(loc='lower right', fontsize=7)#, bbox_to_anchor=(0.5, 0.5))
      
      plt.title("Comparison of Undersampling Algorithms")
      plt.xlabel('Metric')
      plt.ylabel('Result')
      
      for p in ax_eval.patches:
          percentage = '{:.2f}%'.format(100* p.get_height())
          x = p.get_x() + p.get_width()
          y = p.get_height()
          ax_eval.annotate(percentage, (x,y),ha='right', fontsize=8, rotation=90, xytext = (0, -30), textcoords = 'offset points', color="white")    
      
      st.pyplot(fig_eval, clear_figure=True)  

  #---CREATE BARPLOT bars grouped by undersampling algorithm-----
    groupby_algo_expander = st.beta_expander("Comparison Results Grouped by Undersampling Algorithm", expanded=False)
    with groupby_algo_expander:
      plotted_algos = ["Unbalanced",
                   "K-means++ and\n Centroids", 
                   "K-means++ and\n Top1 centroids' \nnearest neighbor", 
                   "K-means++ and \nRandom sampling", 
                   "K-means++ and\n TopN centroids' \nnearest neighbors"]
      
      accuracy = []
      for key in results_dict:
        accuracy.append(results_dict[key]["accuracy"])
        
      precision = []
      for key in results_dict:
        precision.append(results_dict[key]["precision"])
  
      recall = []
      for key in results_dict:
        recall.append(results_dict[key]["recall"])
        
      f1 = []
      for key in results_dict:
        f1.append(results_dict[key]["f1"])
        
      auc = []
      for key in results_dict:
        auc.append(results_dict[key]["auc"])
      
      bars = np.arange(len(accuracy))
      
      fig_eval, ax_eval = plt.subplots()
      plt.bar(bars, accuracy, color = 'C0', width = 0.15, label = "accuracy")
      plt.bar(bars + 0.17, precision, color = 'C1', width = 0.15, label = "precision")
      plt.bar(bars + 0.34, recall, color = 'C2', width = 0.15, label = "recall")
      plt.bar(bars + 0.51, f1, color = 'C3', width = 0.15, label = "f1")
      plt.bar(bars + 0.68, auc, color = 'C4', width = 0.15, label = "auc")
      
      plt.xticks([i + 0.2 for i in range(len(plotted_algos))], plotted_algos, rotation=0, fontsize=7)
      
      ax_eval.legend(loc='lower right', fontsize=7)#, bbox_to_anchor=(0.5, 0.5))
      
      plt.title("Comparison of Undersampling Algorithms")
  
      plt.xlabel('Undersampling Algorithm')
      plt.ylabel('Result')
      
      for p in ax_eval.patches:
          percentage = '{:.2f}%'.format(100* p.get_height())
          x = p.get_x() + p.get_width()
          y = p.get_height()
          ax_eval.annotate(percentage, (x, y),ha='right', fontsize=8, rotation=90, xytext = (0, -30), textcoords = 'offset points', color="white")    
      
      st.pyplot(fig_eval, clear_figure=True)  



######################################## MODE DATA EXPLORATION ###################################################################

if mode == "Data Exploration":
  st.title("Description of " + data_selection)
  with st.sidebar:
    display_countplot(orig_data, target)
  orig_data = orig_data.astype(type_dict)

#---SIZE OF DATASET-----

  st.write("Number of Rows: " + str(len(orig_data)))
  st.write("Number of Columns: " + str(len(orig_data.columns)))
  st.write("Target Variable: " + target)

#---SHOW VARIABLE CATEGORIES AND STATISTICS-----

  category_expander = st.beta_expander("Show variable distributions")

  with category_expander:
    st.write("Column overview:")
    st.write(orig_data.columns)
    cat_cols = []
    st.write("Categorical and Binary Columns:")
    for column in orig_data.columns:
      if orig_data[column].dtype=="O":
          st.write(orig_data[column].value_counts())
          cat_cols.append(column)
    st.write("Numerical Columns:")
    st.write(round(orig_data.describe(),1))
    
#---CREATE BARPLOT 'Target vs. chosen categorical column'    
  st.markdown("""**Conditional Comparison for Target and Categorical/Binary Columns:**""")
  cols = st.beta_columns(2)
  cat_cols.remove(target)
  
  chosen_column = cols[0].radio("Independent Categorical and Binary Columns:", cat_cols, index=0)

  ax = pd.crosstab(orig_data[chosen_column], orig_data[target]).plot(kind="bar")
  ax.set_xlabel(chosen_column)
  ax.set_ylabel("Frequency")
  ax.tick_params(axis='both', which='major')
  fig = ax.get_figure()
  cols[1].pyplot(fig, clear_figure=True)
