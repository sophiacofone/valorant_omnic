# 7/18/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is to run the decision tree models for all data

from class_model import d_tree, f_importance, d_tree_tuning, vis_dtree, prune_dtree

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_decisiontree_process(csv_in):

    df = pd.read_csv(csv_in)

    # Train Test Split
    X = df.drop('self_character', axis=1)
    y = df['self_character']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    cols = X_train.columns

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #d_tree_tuning
    best_dtc = d_tree_tuning(X_train, y_train, X_test, y_test)


def prune_tree(csv_in,csv_out,min_samples_leaf,min_samples_split,max_depth):
    df = pd.read_csv(csv_in)

    # Train Test Split
    X = df.drop('self_character', axis=1)
    y = df['self_character']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    cols = X_train.columns

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pruned = prune_dtree(X_train,y_train,X_test,y_test,min_samples_leaf,min_samples_split,max_depth)

    feature_importances = f_importance(pruned, cols,csv_out)

    return pruned, cols



############# Decision Tree ##############
# run_decisiontree_process('roles/csv/roles_alldf_prepro_data.csv')
# pruned,cols = prune_tree('roles/csv/roles_alldf_prepro_data.csv','df_role_import_dtree_all_20.csv',1,2,20)

# run_decisiontree_process('roles/csv/df_no_userid.csv')
# pruned,cols = prune_tree('roles/csv/df_no_userid.csv','df_role_import_dtree_no_userid_18.csv',1,2,18)

# run_decisiontree_process('roles/csv/df_no_chars.csv')
# pruned,cols = prune_tree('roles/csv/df_no_chars.csv','df_role_import_dtree_no_chars_15.csv',1,2,15)

run_decisiontree_process('roles/csv/df_no_map.csv')
pruned,cols = prune_tree('roles/csv/df_no_map.csv','df_role_import_dtree_no_map_6.csv',1,5,13)

vis_dtree(pruned, cols,'png')
