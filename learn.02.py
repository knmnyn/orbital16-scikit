#! /usr/bin/env python3
from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import svm
from sklearn import metrics
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# ------ START defs
def get_data(s):
    if os.path.exists(s):
        print ("-- FBCI " + s + " data found locally")
        df = pd.read_csv(s)
    else:
        exit("-- Unable to find file")
    return df

def write_tree(tree, feature_names):
    """Create tree dot file for input to graphviz.
    Args
    ----
    tree -- scikit-learn DecisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

def write_predictions(predictions):
    with open("predict.csv","w") as f:
        f.write("id,place_id\n")
        line = 0
        for i in predictions:
            f.write(str(line) + "," + str(i)+' 91 99 1\n')
            line += 1

# ------ END defs

# get the data frame via pandas
train_df = get_data("train.01.csv")

def prepare_data(df, training=False):
    ##time related features (assuming the time = minutes)
    initial_date = np.datetime64('2014-01-01T01:01',   #Arbitrary decision
                                 dtype='datetime64[m]') 
    #working on df_train  
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df.time.values)    
    df['hour'] = d_times.hour
    df['weekday'] = d_times.weekday
    df['day'] = d_times.day
    df['month'] = d_times.month
    df['year'] = d_times.year
    df = df.drop(['time'], axis=1)
    if (training):
        df['place_id2'] = df['place_id']
        df = df.drop(['place_id'], axis=1)
        df['place_id'] = df['place_id2']
        df = df.drop(['place_id2'], axis=1)
    return df

train_df = prepare_data(train_df, training=True)
print ("train_df.head()", train_df.head(), sep="\n", end="\n\n")

sns.set(color_codes=True)
sns.jointplot(x="x", y="y", data=train_df);
sns.jointplot(x="x", y="hour", data=train_df);
sns.jointplot(x="y", y="hour", data=train_df);
sns.plt.show()

# fitting the decision tree with scikit-learn
"""
The decision tree, imported at the start of the post, is initialized
with two parameters: min_samples_split=100 requires 100 samples in a
node for it to be split (this will make more sense when we see the
result) and random_state=99 to seed the random number generator.
"""
start = 1
stop = 9
features = list(train_df.columns[start:stop])
train_y = train_df["place_id"]
train_X = train_df[features]
# Decision Tree
# clf = DecisionTreeClassifier(min_samples_split=50, random_state=99)

# Linear Support Vector Machine
# clf = svm.SVC(kernel='linear', C=1)

# K Nearest Neighbors
# clf = KNeighborsClassifier(3)

# Naive Bayes
# clf = GaussianNB()

## Ensemble Methods
# Random Forest ('Generalized' DT)
# clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
# AdaBoost (also usually based on DTs)
clf = AdaBoostClassifier()

scores = cross_validation.cross_val_score(clf,train_X,train_y, cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# write_tree(dt_model, features) # output the model for graphviz visualization
# train_predicted = dt_model.predict(train_X)

