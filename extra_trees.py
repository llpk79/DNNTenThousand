import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter
from itertools import combinations_with_replacement as combos
from itertools import permutations as perms
from tensorflow.keras import layers, Model
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import coverage_error, f1_score, label_ranking_average_precision_score, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.data import Dataset


# Define game rules.
scoring_rules = [[100, 200, 1000, 2000, 4000, 5000],
                 [0, 0, 200, 400, 800, 5000],
                 [0, 0, 300, 600, 1200, 5000],
                 [0, 0, 400, 800, 1600, 5000],
                 [50, 100, 500, 1000, 2000, 5000],
                 [0, 0, 600, 1200, 2400, 5000]
                 ]


def is_three_pair(choice):
    choice = sorted(choice)
    return (len(choice) == 6 and choice[0] == choice[1] and
            choice[2] == choice[3] and choice[4] == choice[5])


def is_straight(choice):
    return sorted(choice) == list(range(1, 7))


def score_all():
    return [1.] * 6


def choose_dice(roll):
    """Returns label. Dice to be kept are labeled 1.0 else 0.0"""
    counts = Counter(roll)
    if is_three_pair(roll) and (sum(scoring_rules[die - 1][count - 1] for die, count in counts.items()) < 1500):
        choice = score_all()
    elif is_straight(roll):
        choice = score_all()
    else:
        picks = set()
        for die, count in counts.items():
            if scoring_rules[die - 1][count - 1] > 0:
                picks.add(die)
        choice = [0.] * 6
        for i, x in enumerate(roll):
            if x in picks:
                choice[i] = 1.
    return np.array(choice)


def make_some_features(numbers, clip):
    """Make a set of random lists of 6 numbers 1 - 6."""
    features = set()
    combinations = (combo for combo in combos(numbers, 6))
    for i, comb in enumerate(combinations):
        if i % clip == 0:  # Keeping size reasonable
            for perm in perms(comb):
                features.add(perm)
    return features

def create_features_labels():
    features = make_some_features(list(range(1, 7)), 2)

    #Make a numpy array of each list.
    all_features = np.array([np.array(feature) for feature in features])
    #Make a label for each feature.
    all_labels = np.array([choose_dice(feature) for feature in all_features])
    #Ensure arrays are of equal shape
    assert all_features.shape == all_labels.shape
    return all_features, all_labels


def create_dataset(features, labels):
    """Create a column for each die and it's coresponding label by taking each
    column of features and labels."""
    data = {str(i): features[:,i] for i in range(6)}
    dataset = pd.DataFrame(data)
    label = {'{}_l'.format(i): labels[:,i] for i in range(6)}
    label_df = pd.DataFrame(label)
    df = pd.concat([dataset, label_df], axis=1, sort=False)
    return df


def train_val_test_split(X, 
                         y,
                         train_size=.8,
                         val_size=.1,
                         test_size=.1,
                         random_state=42,
                         shuffle=True):
    
    assert train_size + val_size + test_size == 1
        

    X_trainval, X_test, y_trainval, y_test = train_test_split(X,
                                                              y,
                                                              test_size=test_size,
                                                              random_state=random_state,
                                                              shuffle=shuffle,
                                                              stratify=y
                                                             )
    X_train, X_val, y_train, y_val = train_test_split(X_trainval,
                                                      y_trainval,
                                                      test_size=val_size / (train_size + val_size),
                                                      random_state=random_state,
                                                      shuffle=shuffle,
                                                      stratify=y_trainval
                                                     )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train):
    """Train and ExtraTreesClassifier"""
    extra = ExtraTreesClassifier(bootstrap=True,
                                 max_depth=25,
                                 n_jobs=-1,
                                 oob_score=True,
                                 min_samples_split=3,
                                 n_estimators=2250)

    extra.fit(X_train, y_train)


    params = {'min_samples_split': [4, 5, 6],
              'max_depth': [27, 30, 33]}
    grid = GridSearchCV(extra,
                        param_grid=params,
                        scoring='average_precision',
                        n_jobs=-1,
                        cv=5,
                        verbose=1)
    grid.fit(X_train, y_train)


    best = grid.best_estimator_
    
    return best


def pickle_it(model):
    """Serialize trained model."""
    model_b = pickle.dumps(model)
    pickle.dump(model_b, open('model.p', 'wb'))


def main():
    """Make featurs and labels, train and seralize model/datasets."""
    
    print('Creating dataset...')
    all_features, all_labels = create_features_labels()
    df = create_dataset(all_features, all_labels)

    X = df[['0', '1', '2', '3', '4', '5']]
    y = df[['0_l', '1_l', '2_l', '3_l', '4_l', '5_l']]
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    
    assert X_train.shape == y_train.shape
    assert X_val.shape == y_val.shape
    assert X_test.shape == y_test.shape
    
    print('Dataset complete.')
    
    print('Training model...')
    model = train_model(X_train, y_train)
    print('Model complete.')
    
    print('Pickling model...')
    pickle_it(model)
    print('Pickling complete')
    
    print('Saving val and test sets...')
    X_val.to_csv('X_val.csv')
    y_val.to_csv('y_val.csv')
    X_test.to_csv('X_test.csv')
    y_test.to_csv('y_test.csv')
    print('All done.')
    

if __name__ == "__main__":
    main()