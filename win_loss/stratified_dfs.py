import pandas as pd

# Read the CSV file into DataFrame
df = pd.read_csv('win_loss/csv/wl_alldf_prepro_data.csv')

# attack_df = df[df['round_info_ally_side_attacker'] == 1]
# defend_df = df[df['round_info_ally_side_defender'] == 0]
# attack_df.to_csv('wl_alldf_prepro_data_attack.csv', index=False)
# defend_df.to_csv('wl_alldf_prepro_data_defend.csv', index=False)

# cols_to_keep = [col for col in df.columns if "pre_spike" not in col]
# post_spike_df = df[cols_to_keep]
# post_spike_df.to_csv('wl_alldf_prepro_data_post_spike.csv', index=False)

# cols_to_keep = [col for col in df.columns if "post_spike" not in col]
# pre_spike_df = df[cols_to_keep]
# pre_spike_df.to_csv('wl_alldf_prepro_data_pre_spike.csv', index=False)

# cols_to_keep = [col for col in df.columns if "deaths" not in col]
# pre_spike_df = df[cols_to_keep]
# pre_spike_df.to_csv('wl_alldf_prepro_data_no_deaths.csv', index=False)