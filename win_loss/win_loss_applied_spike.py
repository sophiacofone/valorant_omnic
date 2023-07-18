# 7/18/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is to run the decision tree models for the pre-post spike data

from win_loss_model import d_tree, prune_dtree, f_importance, d_tree_tuning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


### Load Data ###
# Read the CSV file into DataFrame
df = pd.read_csv('win_loss/csv/wl_alldf_prepro_data.csv')
attack_df = df[df['round_info_ally_side_attacker'] == 1]
defend_df = df[df['round_info_ally_side_defender'] == 0]


### Train Test Split ###
X_attack = attack_df.drop('round_info_round_won', axis=1)
y_attack = attack_df['round_info_round_won']

X_attack_train, X_attack_test, y_attack_train, y_attack_test = train_test_split(X_attack, y_attack, test_size=0.2, random_state=1)


X_defend = defend_df.drop('round_info_round_won', axis=1)
y_defend = defend_df['round_info_round_won']

X_defend_train, X_defend_test, y_defend_train, y_defend_test = train_test_split(X_defend, y_defend, test_size=0.2, random_state=1)

### Scale the data ###
scaler = StandardScaler()
X_attack_train_scaled = scaler.fit_transform(X_attack_train)
X_attack_test_scaled = scaler.transform(X_attack_test)

X_defend_train_scaled = scaler.fit_transform(X_defend_train)
X_defend_test_scaled = scaler.transform(X_defend_test)

print('attack')
untuned_dtree = d_tree(X_attack_train_scaled,X_attack_test_scaled,X_attack_test,y_attack_test)
# tuned_dtree = d_tree_tuning(X_train, y_train, X_test, y_test)
# maxdepth_dtree = prune_dtree(X_train,y_train,X_test,y_test,6)
feature_df = f_importance(untuned_dtree)
print(feature_df[:50])

print('defend')
untuned_dtree = d_tree(X_defend_train_scaled,X_defend_test_scaled,X_defend_test,y_defend_test)
# tuned_dtree = d_tree_tuning(X_train, y_train, X_test, y_test)
# maxdepth_dtree = prune_dtree(X_train,y_train,X_test,y_test,6)
feature_df = f_importance(untuned_dtree)
print(feature_df[:50])


# fig, ax = plt.subplots(figsize=(20, 20)) 
# tree.plot_tree(maxdepth_dtree, 
#                feature_names=X_train.columns, 
#                class_names=['Loss', 'Win'], 
#                filled=True,
#                rounded=True,
#                ax=ax)

# plt.show()