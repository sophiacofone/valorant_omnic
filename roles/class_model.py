# 7/18/23, Sophia Cofone, Omnic ML Project
# Purpose of these functions are to make decision tree models for the class classification

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 


def d_tree(X_train,y_train,X_test,y_test):
    dtc = DecisionTreeClassifier(random_state=1)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    y_pred_train = dtc.predict(X_train)
    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print(confusion_matrix(y_train, y_pred_train))
    print(classification_report(y_test, y_pred))

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return dtc

def prune_dtree(X_train,y_train,X_test,y_test,min_samples_leaf,min_samples_split,max_depth):
    dtc = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,max_depth=max_depth,random_state=1)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    y_pred_train = dtc.predict(X_train)
    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print(confusion_matrix(y_train, y_pred_train))
    print(classification_report(y_test, y_pred))

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return dtc

# def vis_dtree(dtc, columns,save_path):
#     plt.figure(figsize=(12, 6))
#     plot_tree(dtc, feature_names=columns, class_names=str(dtc.classes_), filled=True)
#     plt.savefig(save_path, dpi=600, bbox_inches='tight')
#     plt.show()

def vis_dtree(dtc, columns,save_path):
    # DOT data
    dot_data = tree.export_graphviz(dtc, out_file=None,
                                    feature_names=columns,  
                                    class_names=['Sentinels', ' Controllers', 'Duelists','Initiators'],
                                    filled=True)

    # Draw graph
    graph = graphviz.Source(dot_data, format=save_path) 
    graph.render("decision_tree")  

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
    print(confusion_matrix(y_train, y_pred_train))
    print(classification_report(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return best_dtc

def f_importance(dtc, columns,csv_name):
    feature_importances = pd.DataFrame(dtc.feature_importances_,
                                   index = columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
    
    feature_importances.to_csv(csv_name)
    return feature_importances
