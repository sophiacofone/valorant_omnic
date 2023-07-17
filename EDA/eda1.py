# 7/14/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is EDA

### Required imports ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### Load Data ###

# Read the CSV files into separate DataFrames
df1 = pd.read_csv('preprocess/csv/pro2_abilities.csv')
df2 = pd.read_csv('preprocess/csv/pro1_abilities.csv')
df3 = pd.read_csv('preprocess/csv/dev_abilities.csv')

# Concatenate the DataFrames vertically
df = pd.concat([df1, df2, df3], ignore_index=True)

### Drop cols I don't need ###
df = df.drop(['player','round_number'], axis=1)

def explore_attributes_bar(df, attribute):
    grouped_df = df.groupby(attribute).size()
    grouped_df.plot(kind='bar')
    plt.xlabel(attribute)
    plt.ylabel('Count')
    plt.title('Grouped Data by {}'.format(attribute))
    plt.show()

# explore_attributes_bar(df, 'spike_planted')
# explore_attributes_bar(df, 'round_info_ally_side')
# explore_attributes_bar(df, 'map')
# explore_attributes_bar(df, 'round_info_round_won')
# explore_attributes_bar(df, 'user_id')

def explore_attributes_wl_ratio(df, attribute):
    win_ratios = df.groupby(attribute)['round_info_round_won'].mean()
    plt.figure(figsize=(10,6))
    sns.barplot(x=win_ratios.index, y=win_ratios.values)
    plt.title('Win Ratio by {}'.format(attribute))
    plt.ylabel('Win Ratio')
    plt.xticks(rotation=90)
    plt.show()

# explore_attributes_wl_ratio(df, 'map')
# explore_attributes_wl_ratio(df, 'self_character')
# explore_attributes_wl_ratio(df, 'spike_planted')
# explore_attributes_wl_ratio(df, 'round_info_ally_side')

def histo_grid(df):
    class_label = 'round_info_round_won'
    columns = ['match_length','seconds_alive','elims','assists','headshots','shielding','first_bloods','wallbangs','credits_earned']
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))
    for i, column in enumerate(columns):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        sns.histplot(data=df, x=column, hue=class_label, ax=ax, bins=50)
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of {column}')
    plt.tight_layout()
    plt.show()

