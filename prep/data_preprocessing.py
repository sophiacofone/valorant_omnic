# 7/14/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is to prepare/pre-process the data for modeling
# Includes dropping un-needed cols, translated boolean cols to 0 and 1, one-hot encoding for categorical attributes

### Required imports ###
import pandas as pd

### Load Data ###
# Read the CSV files into separate DataFrames
df1 = pd.read_csv('prep/csv/pro2_abilities.csv')
df2 = pd.read_csv('prep/csv/pro1_abilities.csv')
df3 = pd.read_csv('prep/csv/dev_abilities.csv')
# Concatenate the DataFrames vertically
df4 = pd.concat([df1, df2, df3], ignore_index=True)

def create_prepro_data(df, csv_title):

    ### Drop cols I don't need ###
    df = df.drop(['player','round_number'], axis=1)

    # Re-mapping booleans
    mapping = {True: 1, False: 0}
    df.loc[:, 'spike_planted'] = df['spike_planted'].map(mapping)
    df.loc[:, 'round_info_round_won'] = df['round_info_round_won'].map(mapping)

    # One-hot
    df_encoded = pd.get_dummies(df, columns=['map', 'ally4_character', 
                                            'ally1_character', 'ally2_character', 'ally3_character', 
                                            'opponent5_character', 'opponent6_character', 'opponent7_character',
                                            'opponent8_character', 'opponent9_character',
                                            'round_info_ally_side','self_longest_inv_state', 
                                            'self_longest_gun_primary','self_longest_gun_secondary',
                                            'self_post_spike_longest_inv_state','self_pre_spike_longest_inv_state',
                                            'self_post_spike_longest_gun_secondary','self_pre_spike_longest_gun_secondary'
                                            ,'self_post_spike_longest_gun_primary','self_pre_spike_longest_gun_primary'])

    # Adding round length feature
    df_encoded['round_info_round_length'] = df_encoded['round_info_round_end'] - df_encoded['round_info_round_start']

    df_encoded.to_csv(csv_title, index=False)

### CREATING THE CSVs ###
create_prepro_data(df1, 'df1_prepro_data.csv')
create_prepro_data(df2, 'df2_prepro_data.csv')
create_prepro_data(df3, 'df3_prepro_data.csv')
create_prepro_data(df4, 'alldf_prepro_data.csv')

