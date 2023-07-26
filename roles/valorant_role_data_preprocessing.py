# 7/14/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is to prepare/pre-process the data for role modeling
# Includes dropping un-needed cols, translated boolean cols to 0 and 1, one-hot encoding for categorical attributes

### Required imports ###
import pandas as pd

### Load Data ###
# Read the CSV files into separate DataFrames
df1 = pd.read_csv('preprocess/csv/pro2_abilities.csv')
df2 = pd.read_csv('preprocess/csv/pro1_abilities.csv')
df3 = pd.read_csv('preprocess/csv/dev_abilities.csv')
# Concatenate the DataFrames vertically
df4 = pd.concat([df1, df2, df3], ignore_index=True)

def create_prepro_data(df, csv_title):

    ### Drop cols I don't need ###
    df = df.drop(['player','round_number'], axis=1)

    # Re-mapping booleans
    mapping = {True: 1, False: 0}
    df.loc[:, 'spike_planted'] = df['spike_planted'].map(mapping)
    df.loc[:, 'round_info_round_won'] = df['round_info_round_won'].map(mapping)
    df.loc[:, 'all_ally_dead'] = df['all_ally_dead'].map(mapping)
    df.loc[:, 'all_opponent_dead'] = df['all_opponent_dead'].map(mapping)

    # One-hot
    df_encoded = pd.get_dummies(df, columns=['map','ally4_character', 
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

    # encoding by valorant class
    mapping = {'killjoy':0, 'cypher':0, 'sage':0,'chamber':0, 
           'brimstone':1, 'omen':1,'viper':1,'astra':1,'harbor':1,
          'jett':2, 'phoenix':2,'raze':2, 'reyna':2,'neon':2, 'yoru':2,
           'gekko':3, 'breach':3, 'fade':3, 'kay/o':3, 'skye':3,'sova':3}
    df_encoded.loc[:, 'self_character'] = df_encoded['self_character'].map(mapping)
    df_encoded
    
    df_encoded.to_csv(csv_title, index=False)

### CREATING THE CSVs ###
create_prepro_data(df1, 'roles_df1_prepro_data.csv')
create_prepro_data(df2, 'roles_df2_prepro_data.csv')
create_prepro_data(df3, 'roles_df3_prepro_data.csv')
create_prepro_data(df4, 'roles_alldf_prepro_data.csv')

