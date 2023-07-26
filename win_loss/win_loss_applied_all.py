# 7/18/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is to run the decision tree models for all data

from win_loss_model import d_tree, prune_dtree, f_importance, d_tree_tuning, plot_feature_importance

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
top_30f = feature_df[:30]
plot_feature_importance(top_30f)


# fig, ax = plt.subplots(figsize=(20, 20)) 
# tree.plot_tree(maxdepth_dtree, 
#                feature_names=X_train.columns, 
#                class_names=['Loss', 'Win'], 
#                filled=True,
#                rounded=True,
#                ax=ax)

# plt.show()