# 7/14/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is EDA

### Required imports ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### Load Data ###

# Read the CSV files into separate DataFrames
df1 = pd.read_csv('prep/csv/pro2_abilities.csv')
df2 = pd.read_csv('prep/csv/pro1_abilities.csv')
df3 = pd.read_csv('prep/csv/dev_abilities.csv')

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

explore_attributes_bar(df, 'spike_planted')
explore_attributes_bar(df, 'round_info_ally_side')
explore_attributes_bar(df, 'map')
explore_attributes_bar(df, 'round_info_round_won')