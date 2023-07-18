import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

def prune_dtree(X_train,y_train,X_test,y_test,max_depth):
    dtc = DecisionTreeClassifier(max_depth = max_depth,min_samples_leaf=1,min_samples_split=5,random_state=1)
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

def f_importance(dtc):
    feature_importances = pd.DataFrame(dtc.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
    return feature_importances

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


### Load Data ###
# Read the CSV file into DataFrame
df = pd.read_csv('win_loss/csv/wl_alldf_prepro_data.csv')

### Train Test Split ###
X = df.drop('round_info_round_won', axis=1)
y = df['round_info_round_won']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

### Scale the data ###
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# untuned_dtree = d_tree(X_train,y_train,X_test,y_test)
# tuned_dtree = d_tree_tuning(X_train, y_train, X_test, y_test)
maxdepth_dtree = prune_dtree(X_train,y_train,X_test,y_test,6)
feature_df = f_importance(maxdepth_dtree)
print(feature_df[:50])

# fig, ax = plt.subplots(figsize=(20, 20)) 
# tree.plot_tree(maxdepth_dtree, 
#                feature_names=X_train.columns, 
#                class_names=['Loss', 'Win'], 
#                filled=True,
#                rounded=True,
#                ax=ax)

# plt.show()