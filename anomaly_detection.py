'''
Credit-Card-Anomaly- Detection
Date:- 4/Aug/2019
Programming Language/Algorithm: Python/SkLearn Classifier
Author:- Suraj Dakua
'''

#import libraries
import numpy as np  #linear algebra library
import pandas as pd  #to read the csv/sqlite data as dataframes.
import time
import matplotlib.pyplot as plt
import seaborn as sns    #used for data visualization
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE   #dimensionality reduction
import warnings
warnings.filterwarnings('ignore')

#import classifier you want to use for the problem statement I choose SVM
from sklearn.model_selection import train_test_split  #split train test data into 75% and 25%.
from imblearn.over_sampling import SMOTE  #over sampling algorithm
from imblearn.under_sampling import NearMiss   # this algorithm is used to split the class in 50/50
from sklearn.metrics import accuracy_score, precision_score, classification_report, roc_auc_score, recall_score, f1_score
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold      #KFold used for cross-validation

#import the CSV file for the anomaly detection.
dataframe = pd.read_csv('/home/ml_rejolut/Desktop/Credit-Card-Detection/creditcard.csv')
dataframe.head()  #returns the top rows of the dataframe 
dataframe.describe()   #returns the descriptive statistics of the table 

#check for null values in the dataframe
dataframe.isnull().sum().max()  #takes array-like object and finds whether the array is empty.

#print the columns in the CSV file
#dataframe.columns()

#print the fraud data items and not-fraud data items.
print('Successfull transactions', round(dataframe['Class'].value_counts()[0]/len(dataframe) * 100, 2), '% of the dataset')
print('Fraud Cases', round(dataframe['Class'].value_counts()[1]/len(dataframe)* 100,2), '% of the dataset')  #round upto 2 decimals.

#as most of the transactions are not fraudulent so our model will predict every transaction as successfull transaction
#I dont want my model to work on assumptions I want it to recognize patterns of fraud transactions
#so letss first visualize the graph of fraud vs no fraud transactions with seaborn.

colors = ['#B22222','#0000FF']  #Note here the hash indicates this value is in hexadecimal.
#sns.countplot('Class', data=dataframe, palette=colors)   
#plt.title('Visualize the transactions \n (0:Successed Transactions || 1:Fraud Transactions)', fontsize = 14)

#we scale the values of time and amount using sklearn preprocessing.
from sklearn.preprocessing import StandardScaler, RobustScaler
std_scaler = StandardScaler()
rob_scaler = RobustScaler()

#outliers are known as deviations from the rest objects.
#we use robust scaler because it is less prone to outliers.
dataframe['scaled_time'] = rob_scaler.fit_transform(dataframe['Time'].values.reshape(-1,1))  
#reshape(-1,1) reshapes the array to single column.
#reshape(1,-1) this will reshapoe the row of the array.
dataframe['scaled_amount'] = rob_scaler.fit_transform(dataframe['Amount'].values.reshape(-1,1))
#axis = 1 or columns, 0 or index   
#inplace = True means do the operation orelse return none.
dataframe.drop(['Time','Amount'], axis = 1, inplace = True)

#scaling is a method in machine learning use to normalize the independent variables or feature of the data.
scaled_time = dataframe['scaled_time']
scaled_amount = dataframe['scaled_amount']
dataframe.drop(['scaled_time','scaled_amount'], axis = 1, inplace=True)
dataframe.insert(0,'scaled_time', scaled_time)
dataframe.insert(1,'scaled_amount', scaled_amount)

X = dataframe.drop('Class', axis=1)
y = dataframe['Class']

#split the original dataframe into equal fraud and no fraud data.
stratified_fold = StratifiedKFold(n_splits=5, random_state=None,shuffle=False)
for train_index, test_index in stratified_fold.split(X, y):
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

#turn the values into array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)
print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

#suffle the data before creating the subsamples.
dataframe = dataframe.sample(frac=1)
# amount of fraud classes 492 rows.
fraud_df = dataframe.loc[dataframe['Class'] == 1]
non_fraud_df = dataframe.loc[dataframe['Class'] == 0][:492]
normal_distributed_df = pd.concat([non_fraud_df, fraud_df ])

# Shuffle dataframe rows
new_dataframe = normal_distributed_df.sample(frac=1, random_state=42)
new_dataframe.head()
print(new_dataframe['Class'].value_counts()/len(new_dataframe))
#now we have our data equally distributed lets visualise the data using seaborn
sns.countplot('Class', data=new_dataframe, palette=colors)
plt.title('Equally distributed classes.', fontsize = 14)
plt.show()

v14_fraud = new_dataframe['V14'].loc[new_dataframe['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))

v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V10 outliers:{}'.format(outliers))

new_dataframe = new_dataframe.drop(new_dataframe[(new_dataframe['V14'] > v14_upper) | (new_dataframe['V14'] < v14_lower)].index)
print('----' * 44)

# -----> V12 removing outliers from fraud transactions
v12_fraud = new_dataframe['V12'].loc[new_dataframe['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))
outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print('V12 outliers: {}'.format(outliers))
print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))
new_dataframe = new_dataframe.drop(new_dataframe[(new_dataframe['V12'] > v12_upper) | (new_dataframe['V12'] < v12_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_dataframe)))
print('----' * 44)


# Removing outliers V10 Feature
v10_fraud = new_dataframe['V10'].loc[new_dataframe['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))
outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print('V10 outliers: {}'.format(outliers))
print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))
new_df = new_dataframe.drop(new_dataframe[(new_dataframe['V10'] > v10_upper) | (new_dataframe['V10'] < v10_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))

import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

X = new_df.drop('Class', axis=1)
y = new_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

from sklearn.linear_model import LogisticRegression
classifiers = {
    "LogisiticRegression": LogisticRegression(),
}

from sklearn.model_selection import cross_val_score
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV
# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
# We automatically get the logistic regression with the best parameters.
log_reg = grid_log_reg.best_estimator_

# Overfitting Case
log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

# Let's Plot LogisticRegression Learning Curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
# Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, method="decision_function")
print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))

log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
def logistic_roc_curve(log_fpr, log_tpr):
    plt.figure(figsize=(12,8))
    plt.title('Logistic Regression ROC Curve', fontsize=16)
    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01,1,0,1])
    
    
logistic_roc_curve(log_fpr, log_tpr)
plt.show()

y_pred = log_reg.predict(X_train)
undersample_y_score = log_reg.decision_function(original_Xtest)
from sklearn.metrics import average_precision_score

undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      undersample_average_precision))
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)

plt.step(recall, precision, color='#004a93', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#48a6ff')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('UnderSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(undersample_average_precision), fontsize=16)

from sklearn.model_selection import train_test_split, RandomizedSearchCV
print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))

# List to append the score and then find the average
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

# Classifier with optimal parameters
# log_reg_sm = grid_log_reg.best_estimator_
log_reg_sm = LogisticRegression()
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)
# Implementing SMOTE Technique 
# Cross Validating the right way
# Parameters
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
for train, test in stratified_fold.split(original_Xtrain, original_ytrain):
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    best_est = rand_log_reg.best_estimator_
    prediction = best_est.predict(original_Xtrain[test])
    
    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))
    precision_lst.append(precision_score(original_ytrain[test], prediction))
    recall_lst.append(recall_score(original_ytrain[test], prediction))
    f1_lst.append(f1_score(original_ytrain[test], prediction))
    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))
    
print('---' * 45)
print('')
print("accuracy: {}".format(np.mean(accuracy_lst)))
print("precision: {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))
print('---' * 45)

labels = ['No Fraud', 'Fraud']
smote_prediction = best_est.predict(original_Xtest)
print(classification_report(original_ytest, smote_prediction, target_names=labels))

sm = SMOTE(ratio='minority', random_state=42)
# This will be the data were we are going to 
Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)

# We Improve the score by 2% points approximately 
# Implement GridSearchCV and the other models
# Logistic Regression
t0 = time.time()
log_reg_sm = grid_log_reg.best_estimator_
log_reg_sm.fit(Xsm_train, ysm_train)
t1 = time.time()

from sklearn.metrics import confusion_matrix
# Logistic Regression fitted using SMOTE technique
y_pred_log_reg = log_reg_sm.predict(X_test)
log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
fig, ax = plt.subplots(2, 2,figsize=(22,12))
sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)
ax[0, 0].set_title("Logistic Regression Confusion Matrix", fontsize=14)
ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)
plt.show()

from sklearn.metrics import classification_report
print('Logistic Regression:')
print(classification_report(y_test, y_pred_log_reg))









