# 7/18/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is to run the decision tree models for all data

from win_loss_model import d_tree, prune_dtree, f_importance, d_tree_tuning, log_reg_tuning_l1, log_reg_select_f_l1, log_reg_train_l2, log_reg_get_f_l2, vis_dtree, confusion_vis

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
    l2_model, confusion_matrix_test = log_reg_train_l2(C_values, X_train_l1, X_test_l1, y_train, y_test)

    confusion_vis(confusion_matrix_test,'confusion_mat')

    log_reg_get_f_l2(l2_model,selected_features, csv_out)

def run_decisiontree_process(csv_in):

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

    #d_tree_tuning
    best_dtc = d_tree_tuning(X_train, y_train, X_test, y_test)


def prune_tree(csv_in,csv_out,min_samples_leaf,min_samples_split,max_depth):
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

    pruned, confusion_matrix_test = prune_dtree(X_train,y_train,X_test,y_test,min_samples_leaf,min_samples_split,max_depth)

    feature_importances = f_importance(pruned, cols,csv_out)

    confusion_vis(confusion_matrix_test,'confusion_mat')

    return pruned, cols

############# Logistic Regression ##############
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data.csv','df_coefs_logreg_all.csv')
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data_attack.csv','df_coefs_logreg_attack.csv')
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data_defend.csv','df_coefs_logreg_defend.csv')
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data_pre_spike.csv','df_coefs_logreg_prespike.csv')
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data_post_spike.csv','df_coefs_logreg_postspike.csv')
# run_logreg_process('win_loss/csv/wl_alldf_prepro_data_no_deaths.csv','df_coefs_logreg_no_deaths.csv')


############# Decision Tree ##############
# run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data.csv')
# pruned,cols = prune_tree('win_loss/csv/wl_alldf_prepro_data.csv','df_import_dtree_all_5.csv',1,5,5)
# vis_dtree(pruned,cols,"decision_tree_all_5.png")

# run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data_attack.csv')
# pruned,cols = prune_tree('win_loss/csv/wl_alldf_prepro_data_attack.csv','df_import_dtree_attack_5.csv',3,2,5)
# vis_dtree(pruned,cols,"decision_tree_attack_5.png")

# run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data_defend.csv')
# pruned,cols = prune_tree('win_loss/csv/wl_alldf_prepro_data_defend.csv','df_import_dtree_defend__5.csv',3,2,5)
# vis_dtree(pruned,cols,"decision_tree_defend_5.png")

# run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data_pre_spike.csv')
# pruned,cols = prune_tree('win_loss/csv/wl_alldf_prepro_data_pre_spike.csv','df_import_dtree_pres_5.csv',3,2,5)
# vis_dtree(pruned,cols,"decision_tree_pres_5.png")

# run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data_post_spike.csv')
# pruned,cols = prune_tree('win_loss/csv/wl_alldf_prepro_data_post_spike.csv','df_import_dtree_posts_5.csv',3,2,5)
# vis_dtree(pruned,cols,"decision_tree_posts_5.png")

# run_decisiontree_process('win_loss/csv/wl_alldf_prepro_data_no_deaths.csv')
# pruned,cols = prune_tree('win_loss/csv/wl_alldf_prepro_data_no_deaths.csv','df_import_dtree_no_deaths_5.csv',3,10,5)
# vis_dtree(pruned,cols,"decision_tree_no_deaths_5.png")

# run_decisiontree_process('win_loss/csv/sentinels_df.csv')
# pruned,cols = prune_tree('win_loss/csv/sentinels_df.csv','df_import_dtree_sentinels_df_4.csv',2,2,4)
# vis_dtree(pruned,cols,"decision_tree_sentinels_df_4.png")

# run_decisiontree_process('win_loss/csv/controllers_df.csv')
# pruned,cols = prune_tree('win_loss/csv/controllers_df.csv','df_import_dtree_controllers_df_4.csv',1,10,4)
# vis_dtree(pruned,cols,"decision_tree_controllers_df_4.png")

# run_decisiontree_process('win_loss/csv/duelists_df.csv')
# pruned,cols = prune_tree('win_loss/csv/duelists_df.csv','df_import_dtree_duelists_df_4.csv',3,2,4)
# vis_dtree(pruned,cols,"decision_tree_duelists_df_4.png")

# run_decisiontree_process('win_loss/csv/initiators_df.csv')
# pruned,cols = prune_tree('win_loss/csv/initiators_df.csv','df_import_dtree_initiators_df_4.csv',3,2,4)
# vis_dtree(pruned,cols,"decision_tree_initiators_df_4.png")