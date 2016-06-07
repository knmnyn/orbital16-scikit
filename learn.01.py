#! /usr/bin/env python3
from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

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
model = DecisionTreeClassifier(min_samples_split=50, random_state=99)

model.fit(train_X, train_y) # learn the model
write_tree(model, features) # output the model for graphviz visualization
train_predicted = model.predict(train_X)

# output training performance
print(metrics.classification_report(train_y, train_predicted))
print(metrics.confusion_matrix(train_y, train_predicted))

# test
test_df = get_data("test.01.csv")
test_df = prepare_data(test_df)
print ("test_df.head()", test_df.head(), sep="\n", end="\n\n")

test_X = test_df[list(test_df.columns[start:stop+1])]
test_predicted = model.predict(test_X)
print (test_predicted)
write_predictions(test_predicted)
