# 7/18/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is to run the decision tree models for all data

from win_loss_model import d_tree, prune_dtree, f_importance, d_tree_tuning, log_reg_tuning_l1, log_reg_select_f_l1, log_reg_train_l2, log_reg_get_f_l2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_logreg_process(csv_in, csv_out):

    df = pd.read_csv(csv_in)

    # Train Test Split
    X = df.drop('round_info_round_won', axis=1)
    y = df['round_info_round_won']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyper-param tuning grid
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]

    # tuning l1 log reg with CV for C param
    best_l1_model = log_reg_tuning_l1(X_train_scaled, y_train, C_values)

    # feature selecting using l1 logreg
    X_train_l1, X_test_l1,selected_features = log_reg_select_f_l1(X_train, X_train_scaled, X_test_scaled, best_l1_model)

    # new model with selected feats using l2
    l2_model = log_reg_train_l2(C_values, X_train_l1, X_test_l1, y_train, y_test)

    log_reg_get_f_l2(l2_model,selected_features, csv_out)

def run_decisiontree_process(csv_in, csv_out):

    df = pd.read_csv(csv_in)

    # Train Test Split
    X = df.drop('round_info_round_won', axis=1)
    y = df['round_info_round_won']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    cols = X_train.columns
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # #d_tree_tuning
    # best_dtc = d_tree_tuning(X_train, y_train, X_test, y_test)

    dtc = d_tree(X_train_scaled,y_train,X_test_scaled,y_test)

    feature_importances = f_importance(dtc, cols,csv_out)


############# Logistic Regression ##############
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data.csv','df_coefs_logreg_all.csv')
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data_attack.csv','df_coefs_logreg_attack.csv')
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data_defend.csv','df_coefs_logreg_defend.csv')
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data_pre_spike.csv','df_coefs_logreg_prespike.csv')
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data_post_spike.csv','df_coefs_logreg_postspike.csv')
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data_no_deaths.csv','df_coefs_logreg_no_deaths.csv')


############# Decision Tree ##############
run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data.csv','df_coefs_dtree_all.csv')
# run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data_attack.csv','df_coefs_logreg_attack.csv')
# run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data_defend.csv','df_coefs_logreg_defend.csv')
# run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data_pre_spike.csv','df_coefs_logreg_prespike.csv')
# run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data_post_spike.csv','df_coefs_logreg_postspike.csv')
# run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data_no_deaths.csv','df_coefs_logreg_no_deaths.csv')