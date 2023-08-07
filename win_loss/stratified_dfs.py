from socket import send_fds
import pandas as pd

# Read the CSV file into DataFrame
df = pd.read_csv('win_loss/csv/wl_alldf_prepro_data.csv')

df_no_map = pd.read_csv('win_loss/csv/df_no_map.csv')

# sentinels_df = df_no_map[df_no_map['self_character'] == 0]
# sentinels_df.to_csv('sentinels_df.csv', index=False)
# controllers_df = df_no_map[df_no_map['self_character'] == 1]
# controllers_df.to_csv('controllers_df.csv', index=False)
# duelists_df = df_no_map[df_no_map['self_character'] == 2]
# duelists_df.to_csv('duelists_df.csv', index=False)
# initiators_df = df_no_map[df_no_map['self_character'] == 3]
# initiators_df.to_csv('initiators_df.csv', index=False)

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