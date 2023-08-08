# 7/18/23, Sophia Cofone, Omnic ML Project
# Purpose of these functions are to make decision tree models for the W/L classification

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree
import seaborn as sns


def d_tree(X_train,y_train,X_test,y_test):
    dtc = DecisionTreeClassifier(random_state=1)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    y_pred_train = dtc.predict(X_train)
    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Train F1:", f1_score(y_train, y_pred_train))
    print(confusion_matrix(y_train, y_pred_train))
    print(classification_report(y_test, y_pred_train))

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test F1:", f1_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    
    return dtc

def prune_dtree(X_train,y_train,X_test,y_test,min_samples_leaf,min_samples_split,max_depth):
    dtc = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,max_depth=max_depth,random_state=1)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    y_pred_train = dtc.predict(X_train)
    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Train F1:", f1_score(y_train, y_pred_train))
    print(confusion_matrix(y_train, y_pred_train))
    print(classification_report(y_test, y_pred))

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test F1:", f1_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    confusion_matrix_test = confusion_matrix(y_test, y_pred)
    
    return dtc, confusion_matrix_test

def vis_dtree(dtc, columns,save_path):
    plt.figure(figsize=(12, 6))
    plot_tree(dtc, feature_names=columns, class_names=str(dtc.classes_), filled=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

def confusion_vis(confusion_matrix_test,save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_test, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['True Negative', 'True Positive'])

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

def d_tree_tuning(X_train, y_train, X_test, y_test):
    param_grid = {
        'min_samples_leaf': [1, 2, 3],  # Example values for min_samples_leaf
        'min_samples_split': [2, 5, 10]  # Example values for min_samples_split
    }

    dtc = DecisionTreeClassifier(random_state=1)
    grid_search = GridSearchCV(dtc, param_grid, cv=5, verbose=2)
    grid_search.fit(X_train, y_train)
    best_dtc = grid_search.best_estimator_
    y_pred_train = best_dtc.predict(X_train)
    y_pred = best_dtc.predict(X_test)

    print("Best Parameters:", grid_search.best_params_)
    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Train F1:", f1_score(y_train, y_pred_train))
    print(confusion_matrix(y_train, y_pred_train))
    print(classification_report(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test F1:", f1_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return best_dtc

def f_importance(dtc, columns,csv_name):
    feature_importances = pd.DataFrame(dtc.feature_importances_,
                                   index = columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
    
    feature_importances.to_csv(csv_name)
    return feature_importances

def log_reg_tuning_l1(X_train, y_train, C_values):
    # apply L1 regularization (Lasso) for feature selection
    l1_model = LogisticRegression(penalty='l1', solver='liblinear')
    grid_search = GridSearchCV(l1_model, param_grid={'C': C_values}, cv=5, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"Best C for L1 model: {grid_search.best_params_}")

    best_l1_model = grid_search.best_estimator_
    return best_l1_model

def log_reg_select_f_l1(X_train_orig, X_train, X_test, best_l1_model):
    # aelect features whose coefficients are in the top half
    model = SelectFromModel(best_l1_model, threshold='median')
    X_train_l1 = model.transform(X_train)
    X_test_l1 = model.transform(X_test)

    selected_features = [f for f, s in zip(X_train_orig.columns, model.get_support()) if s]
    print(f"Selected features: {selected_features}")

    return X_train_l1, X_test_l1, selected_features

def log_reg_train_l2(C_values, X_train_l1, X_test_l1, y_train, y_test):
    # retrain new model using L2 regularization (Ridge) on the selected features
    l2_model = LogisticRegression(penalty='l2', solver='lbfgs')
    grid_search = GridSearchCV(l2_model, param_grid={'C': C_values}, cv=5)
    grid_search.fit(X_train_l1, y_train)

    print(f"Best C for L2 model: {grid_search.best_params_}")

    best_C = grid_search.best_params_['C']
    l2_model = LogisticRegression(penalty='l2', solver='lbfgs', C=best_C)
    l2_model.fit(X_train_l1, y_train)

    # make predictions and evaluate model performance
    y_pred_train_l2 = l2_model.predict(X_train_l1)
    y_pred_test_l2 = l2_model.predict(X_test_l1)
    
    print("Train Accuracy:", accuracy_score(y_train, y_pred_train_l2))
    print("Train F1:", f1_score(y_train, y_pred_train_l2))
    print(confusion_matrix(y_train, y_pred_train_l2))
    
    print(classification_report(y_train, y_pred_train_l2))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test_l2))
    print("Test F1:", f1_score(y_test, y_pred_test_l2))
    print(confusion_matrix(y_test, y_pred_test_l2))
    print(classification_report(y_test, y_pred_test_l2))

    confusion_matrix_test = confusion_matrix(y_test, y_pred_test_l2)

    return l2_model, confusion_matrix_test

def log_reg_get_f_l2(l2_model, selected_features,csv_name):
    # Get the coefficients from the L2 model
    coefs = l2_model.coef_[0]

    # Create a DataFrame of features and coefficients
    df_coefs = pd.DataFrame({'feature': selected_features, 'coefficient': coefs})

    # Sort by absolute value of coefficient
    df_coefs['abs_coef'] = df_coefs['coefficient'].abs()
    df_coefs = df_coefs.sort_values('abs_coef', ascending=False)

    print(df_coefs[['feature', 'coefficient']])

    # Save the coefficients DataFrame to a CSV file
    df_coefs.to_csv(csv_name, index=False)