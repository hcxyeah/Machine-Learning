#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from sklearn.feature_selection import SelectPercentile
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import StratifiedShuffleSplit


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'total_payments', 'bonus', 'expenses','other',
                 'deferred_income','total_stock_value', 'long_term_incentive', 'restricted_stock','loan_advances',
                 'exercised_stock_options'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print len(data_dict.keys())
poi = 1
for i in data_dict.keys():
    if data_dict[i]["poi"] == 1:
        poi += 1
print poi



### Task 2: Remove outliers

data_dict.pop("TOTAL", 0) #remove "total"
data_dict.pop("LOCKHART EUGENE E", 0) # this person missing all value

data_dict["BELFER ROBERT"]["deferred_income"] = -102500
data_dict["BELFER ROBERT"]["deferral_payments"] = "NaN"
data_dict["BHATNAGAR SANJAY"]["restricted_stock"] = 2604490
data_dict["BELFER ROBERT"]["restricted_stock_deferred"] = -44093

### Task 3: Create new feature(s)


### Store to my_dataset for easy export below.
my_dataset = data_dict


for name in my_dataset.keys():
    if my_dataset[name]["from_poi_to_this_person"] == 'NaN' or my_dataset[name]["from_this_person_to_poi"] == 'NaN' or\
                    my_dataset[name]['shared_receipt_with_poi'] == 'NaN' or my_dataset[name]["to_messages"] == 'NaN' or \
                    my_dataset[name]["from_messages"] == 'NaN':
        fraction_from_to_cc_pois = 0
    else:
        from_to_cc_pois = my_dataset[name]["from_poi_to_this_person"] + my_dataset[name]["from_this_person_to_poi"] + \
                      my_dataset[name]['shared_receipt_with_poi']
        all_messages = my_dataset[name]["to_messages"] + my_dataset[name]["from_messages"]
        fraction_from_to_cc_pois = float(from_to_cc_pois) / all_messages
    my_dataset[name]["fraction_from_to_cc_pois"] = fraction_from_to_cc_pois

features_list.append("fraction_from_to_cc_pois")

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN = True) #convert dictionary to np array
labels, features = targetFeatureSplit(data)

cv = StratifiedShuffleSplit(labels, 1000, random_state = 42, test_size=0.3)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx, test_idx in cv:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import grid_search
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif

def calculate_precision_recall(pred, labels_test):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in pred:
        for j in labels_test:
            if i == 1 and j == 1:
                true_positive += 1
            if i == 1 and j == 0:
                false_positive += 1
            if i == 0 and j == 1:
                false_negative += 1
    print "precision:"
    if true_positive + false_positive == 0:
        precision = 0
    else:
        precision = float(true_positive)/(true_positive+false_positive)
    print precision
    print "recall:"
    if true_positive + false_negative == 0:
        recall = 0
    else:
        recall = float(true_positive)/(true_positive+false_negative)
    print recall


#features_train, features_test, labels_train, labels_test = \
 #       train_test_split(features, labels, test_size=0.3, random_state=42)
def use(method):
    if method == 'naive bayes':
        estimators = [("skb", SelectKBest(score_func=f_classif)),('pca', PCA()),
                      ('bayes',GaussianNB())]
        clf = Pipeline(estimators)
        parameters = {"skb__k":[8,9,10,11,12],
                      "pca__n_components":[2,6,4,8]}
        clf = grid_search.GridSearchCV(clf, parameters)
        scaler = MinMaxScaler()
        features_train_scaled = scaler.fit_transform(features_train)
        features_test_scaled = scaler.transform(features_test)
        clf.fit(features_train_scaled, labels_train)
        pred = clf.predict(features_test_scaled)
        print clf.best_params_
        features_k = clf.best_params_['skb__k']
        SKB_k = SelectKBest(f_classif, k = features_k)
        SKB_k.fit_transform(features_train_scaled, labels_train)
        print "features score: "
        print SKB_k.scores_
        features_selected = [features_list[1:][i]for i in SKB_k.get_support(indices=True)]
        print features_selected
    elif method == 'svm':
        estimators = [('reduce_dim', PCA()), ('svc', SVC())]
        clf = Pipeline(estimators)
        parameters = {'svc__C': [1,10]}
        clf = grid_search.GridSearchCV(clf, parameters)
        scaler = MinMaxScaler()
        features_train_scaled = scaler.fit_transform(features_train)
        features_test_scaled = scaler.transform(features_test)
        clf.fit(features_train_scaled, labels_train)
        pred = clf.predict(features_test_scaled)
        print clf.best_estimator_
    elif method == 'decision tree':
        estimators = [("skb", SelectKBest(score_func=f_classif)),('pca', PCA()),
                      ('tree', tree.DecisionTreeClassifier())]
        clf = Pipeline(estimators)
        parameters = {"tree__min_samples_split": [2,10],"skb__k":[8,9,10,11,12],
                      "pca__n_components":[2,4,6,8]}
        clf = grid_search.GridSearchCV(clf, parameters)
        scaler = MinMaxScaler()
        features_train_scaled = scaler.fit_transform(features_train)
        features_test_scaled = scaler.transform(features_test)
        clf.fit(features_train_scaled, labels_train)
        pred = clf.predict(features_test_scaled)
        print clf.best_params_
        features_k = clf.best_params_['skb__k']
        SKB_k = SelectKBest(f_classif, k = features_k)
        SKB_k.fit_transform(features_train, labels_train)
        features_selected = [features_list[1:][i]for i in SKB_k.get_support(indices=True)]
        print features_selected
    accuracy = accuracy_score(labels_test, pred)
    print "accuracy score:"
    print accuracy
    calculate_precision_recall(pred, labels_test)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


print "    naive bayes:"
use("naive bayes")
print "    svm:"
use("svm")
print "    decision tree"
use("decision tree")

## Use the model
features_list = ['poi', 'salary', 'total_payments', 'bonus', 'expenses','fraction_from_to_cc_pois',
                 'deferred_income','total_stock_value', 'long_term_incentive','loan_advances',
                 'exercised_stock_options'] # You will need to use more features

#estimators = [('pca', PCA(n_components=2)), ('nb', GaussianNB())]
estimators = [('pca', PCA(n_components=2)), ('tree', tree.DecisionTreeClassifier(min_samples_split=10))]
clf = Pipeline(estimators)

## calculate precision and recall


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
