# 7/14/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is re-map the abilities according to the valorant_abilities sheet
# Idea is to provide more context/information regarding what the ability does based on a specific character's kit
# aka feature engineering

### Required imports ###
import pandas as pd

def map_abilities(json_filename,csv_title):
    df = pd.read_csv(json_filename)
    abilities_df = pd.read_csv('prep/abilities.csv')

    abilities_df['agent_name'] = abilities_df['agent_name'].str.lower()
    # Mapping dictionary
    mapping = {'1': 1, '2': 2, '3': 3, 'ultimate': 4}
    # Replace values using the mapping
    abilities_df['ability'] = abilities_df['ability'].replace(mapping)

    df_copy = df.copy()

    # Create a set of the abilities characteristics columns
    ability_cols = set(['crowd_control_general', 'crowd_control_mobility', 'crowd_control_vision', 
                        'damage_for_team', 'damage_for_self', 'shield', 'information', 'healing'])

    # Initialize new columns to 0
    for ability_col in ability_cols:
        df_copy[f"pre_spike_{ability_col}_used"] = 0
        df_copy[f"post_spike_{ability_col}_used"] = 0

    # Iterate over the rows of the new_df dataframe
    for idx, row in df_copy.iterrows():
        agent = row['self_character']
        
        for col in df_copy.columns:
            if 'ability_usage' in col:
                # Determine if the ability was used pre or post spike
                prepost = 'pre' if 'pre' in col else 'post'
                ab_num = int(col.split('ability_usage_')[1])

                # Only proceed if the ability was used
                if row[col] > 0:
                    # Get the specific row from abilities_df
                    agent_abilities_row = abilities_df[(abilities_df['agent_name'] == agent) & (abilities_df['ability'] == ab_num)]

                    # For each ability characteristic column in the abilities dataframe
                    for ability_col in ability_cols:
                        # If the ability has the characteristic, add a column in the main dataframe
                        if agent_abilities_row[ability_col].values[0] == 1:
                            new_col_name = f"{prepost}_spike_{ability_col}_used"
                            # Assign the number of times the ability was used to the new column
                            df_copy.at[idx, new_col_name] = row[col]
    
    df_copy.to_csv(csv_title, index=False)
        
map_abilities('parsing/csv/dev_data_norm_round_spikesplit.csv','dev_abilities.csv')
map_abilities('parsing/csv/pro1_data_norm_round_spikesplit.csv','pro1_abilities.csv')
map_abilities('parsing/csv/pro2_data_norm_round_spikesplit.csv','pro2_abilities.csv')