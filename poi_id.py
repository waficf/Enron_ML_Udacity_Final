#!/usr/bin/python

import sys
import pickle
import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


############################ Explore the data ############################

# Convert data into a pandas dataframe
data_df = pd.DataFrame.from_dict(data_dict, orient='index')

# Drop email column as it is useless for this analysis
data_df = data_df.drop('email_address', axis=1)



print "Total number of data points:", len(data_df)

print "Number of POIs in Dataset:", len(data_df[data_df.poi == 1])

print "Number of Non POIs in Dataset:", len(data_df[data_df.poi == 0])

print "Number of features used:", len(data_df.columns)


# Detect NaNs in every feature available

# Convert columns data type to float to detect NaNs
data_df[['salary','to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',\
'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi','director_fees',\
'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']] = \
data_df[['salary','to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',\
'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi','director_fees',\
'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']].astype(float)

# print data_df.info()

# Getting to better undertsand the dataset
feature_decription = data_df.describe()
# print feature_decription
# feature_decription.to_csv("asd.csv")

data_df[['salary','to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',\
'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi','director_fees',\
'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']] = \
data_df[['salary','to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',\
'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi','director_fees',\
'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']].fillna(0)

# print data_df.info()

# print data_df.head()

### Task 2: Remove outliers

# Plot the salary and bonus
plt.scatter(data_df.salary, data_df.bonus)
plt.title("Salary vs Bonus Before Outlier")
# plt.show()

# Detect the outliers on salary
# print data_df[data_df.salary > 25000000]

# Drop the Total row detected above
data_df = data_df.drop('TOTAL')

# Dropping the THE TRAVEL AGENCY IN THE PARK
data_df = data_df.drop('THE TRAVEL AGENCY IN THE PARK')

# Replot the salary and bonus without Total
plt.scatter(data_df.salary, data_df.bonus)
plt.title("Salary vs Bonus After Outlier")
# plt.show()

q1 = data_df.quantile(0.25)
q3 = data_df.quantile(0.75)

IQR = q3-q1

outliers = data_df[(data_df<(q1 - 1.5*IQR)) | (data_df>(q3 + 1.5*IQR))].count(axis=1)

print outliers.sort_values(ascending=False).head(10)

# Check list of POIs
# print data_df[data_df.poi == 1].poi

# Drop outliers whom are no POIs
# print len(data_df)
data_df = data_df.drop(['FREVERT MARK A', 'LAVORATO JOHN J', 'BAXTER JOHN C'])
# print len(data_df)

# Return the pandas dataframe ot dictionary to be capable of running it with the tester 
my_dataset = data_df.to_dict(orient='index')

# Initial features to be tested
features_list = ['poi','salary','to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',\
'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi','director_fees',\
'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


#### Here I am running the above original features with the 4 models selected
# Gaussian Naive Bayes Classifier
# clf = GaussianNB()

# Decision Tree Classifier
# clf = DecisionTreeClassifier()

# Random Forest Classifier
# clf = RandomForestClassifier()

# Logisitics Regression
# clf = LogisticRegression()


### Task 3: Create new feature(s)


# Feature 1: Feature that gets the ratio of email to POI from the total number of messages
for key, value in my_dataset.items():
	if my_dataset[key]['from_messages'] == 0:
		my_dataset[key]["to_poi_ratio"] = 0
	else:
		my_dataset[key]["to_poi_ratio"] =  float(my_dataset[key]['from_this_person_to_poi'])/float(my_dataset[key]['from_messages'])

# Feature 2: Feature that gets the ratio of email from POI from the total number of messages
for key, value in my_dataset.items():
	if my_dataset[key]['from_messages'] == 0:
		my_dataset[key]["from_poi_ratio"] = 0
	else:
		my_dataset[key]["from_poi_ratio"] =  float(my_dataset[key]['from_poi_to_this_person'])/float(my_dataset[key]['to_messages'])


# Feature 3: Feature that gets the ratio of shared email with a POI
for key, value in my_dataset.items():
	if my_dataset[key]["shared_receipt_with_poi"] == 0 or my_dataset[key]["to_messages"] == 0:
		my_dataset[key]['ratio_cced_poi'] = 0
	else:
		my_dataset[key]['ratio_cced_poi'] = my_dataset[key]['shared_receipt_with_poi']/my_dataset[key]['to_messages']


# Feature 4: Ratio of bonus from salary
for key, value in my_dataset.items():
	if my_dataset[key]['bonus'] == 0 or my_dataset[key]['salary'] == 0:
		my_dataset[key]["ratio_bonus_salary"] = 0
	else: 
		my_dataset[key]["ratio_bonus_salary"] =  float(my_dataset[key]['bonus'])/float(my_dataset[key]['salary'])


# Feature 6: Ratio of bonus from total payments
for key, value in my_dataset.items():
	if my_dataset[key]['bonus'] == 0 or my_dataset[key]['total_payments'] == 0:
		my_dataset[key]["ratio_bonus_payments"] = 0
	else: 
		my_dataset[key]["ratio_bonus_payments"] = float(my_dataset[key]['bonus'])/float(my_dataset[key]['total_payments'])


# New feature list after adding the new 4 feaures
features_list = ['poi','salary', 'bonus', 'to_poi_ratio', 'from_poi_ratio', 'ratio_bonus_salary', 'ratio_bonus_payments','to_messages', 'deferral_payments', \
'total_payments', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi','restricted_stock_deferred', 'total_stock_value', \
'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi','director_fees',\
'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

#### After addign new features and prior any feature reduction or selection, I will run the above models to see if adding the features improved the scores

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Gaussian Naive Bayes Classifier
# clf = GaussianNB()

# Decision Tree Classifier
# clf = DecisionTreeClassifier()

# Random Forest Classifier
# clf = RandomForestClassifier()

# Logisitics Regression
# clf = LogisticRegression()


poi = data[:,0]
salary = data[:,1]
bonus = data[:,2]
to_poi_ratio = data[:,3]
from_poi_ratio = data[:,4]
ratio_bonus_salary = data[:,5]
ratio_bonus_payments = data[:,6]


# Plotting some new and old features
plt.scatter(salary, bonus, c = poi)
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.title("Salary vs Bonus Labeled by POI")
# plt.show()

plt.scatter(to_poi_ratio, from_poi_ratio, c = poi)
plt.xlabel("Ratio of Email to a POI")
plt.ylabel("Ratio of Email from a POI")
plt.title("Ratio of Email From & To a POI")
# plt.show()

plt.scatter(ratio_bonus_salary, ratio_bonus_payments, c = poi)
plt.xlabel("Bonus Ratio from Salary")
plt.ylabel("Bonus Ratio from Total Payments")
plt.title("Bonus Ratio from Salary vs Total Payment")
# plt.show()

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import grid_search
from pprint import pprint
from tester import dump_classifier_and_data, test_classifier


N_FEATURES_OPTIONS = numpy.arange(10, len(features_list))

pipe = Pipeline([
    ('reduce_dim', SelectKBest()),
    ('classify', DecisionTreeClassifier(random_state = 42))
])


param_grid = [
    {
        'reduce_dim__k': N_FEATURES_OPTIONS
    }
]

gs = grid_search.GridSearchCV(estimator = pipe, cv=10, param_grid=param_grid)
gs.fit(features, labels)
print(gs.best_estimator_)


print "Best parameters from parameter grid:"
print gs.best_params_
best_param = gs.best_params_.get('reduce_dim__k')

# get Best estimator
# clf = gs.best_estimator_


skbest = SelectKBest(k = best_param)
sk_transform = skbest.fit(features, labels)
mask = skbest.get_support(True)

for i in mask:
    print '%s score: %f' % (features_list[i + 1], skbest.scores_[i])



### Task 5: Tune your classifier to achieve better than .3 precision and recall 

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.4, random_state=42)

tree_pipe = Pipeline([
    ('select_features', SelectKBest(k = best_param)),
    ('classify', DecisionTreeClassifier(random_state = 22)),
])

param_grid = dict(classify__min_samples_leaf =  [1, 2, 3, 4, 5, 10, 20], 
                  classify__min_samples_split = [2, 8, 10, 20],
                  classify__max_depth = [None, 2, 4, 8, 10, 15],
                  classify__criterion =  ['gini', 'entropy'])

tree_clf = grid_search.GridSearchCV(estimator = tree_pipe, param_grid = param_grid, scoring='f1', cv=10)
    
tree_clf.fit(features_train, labels_train)
    
pred = tree_clf.predict(features_test)

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
acc = accuracy_score(pred, labels_test)
prec = precision_score(pred, labels_test)
recal = recall_score(pred, labels_test)
f1 = f1_score(pred, labels_test)
print "accuracy_score:", acc, "Precision: ", prec, "Recall: ", recal, "f1_score: ", f1


tree_clf.fit(features, labels)
clf = tree_clf.best_estimator_

# print clf

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, features_list)







