# 7/14/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is to parse & normalize the data by round pre and post spike plant
# also includes some data cleaning feature engineering
# Works with dev_data, pro1_data, and pro2_data
# Code is pretty manual/hardcoded. If the strucutre of the data changes or features are added, the code will need to be updated/reviewed.

### Required imports ###
import pandas as pd

def create_norm_csv_spike(json_filename,csv_title):
    ### Create df ###
    df = pd.read_json(json_filename)

    ### Drop the all cols besides 'statistics' and 'user_id' ###
    # These cols either have redundant information, or information that isn't relevant for this analysis
    df_stats = df[['user_id','statistics']]

    ### Flattening the stats col by 1 level ###
    flat1_df_stats = pd.json_normalize(df_stats['statistics'], max_level=0)
    flat1_df_stats = pd.concat([df_stats['user_id'], flat1_df_stats], axis=1)

    ### Removing some cols we don't care about from the ROUND perspective ###
    # Keep in mind the 'totals', 'result' are from the MATCH perspective
    # Even though the map is from match perspective I need it so I am adding it in (similar to userID)
    # This weapon stuff is from match perspective (I gather other weapon info from round data)
    # I think its possible to do something with 'allies_onscreen', 'opponents_onscreen', and 'detections_totals' but I am dropping for now
    flat1_df_stats = flat1_df_stats.drop(['totals','score', 'allies','result', 'status','gametype','version','end_time','opponents','processed','ally_score','start_time','opponent_score','detections_totals','best_weapon_elims','most_used_primary_seconds','analysis_processed','most_used_secondary_seconds','best_weapon_type_elims','opponents_onscreen','allies_onscreen','analysis_status','player_totals','best_weapon_type','best_weapon','most_used_secondary','most_used_primary','best_weapon_type_elims'], axis=1)

    ### Flattening the player_ids col by 1 level ###
    # This is getting all the teammates and opponents, as well as mapping the 'player' to an ally
    # Create an empty list to hold the expanded player ids
    expanded_player_ids_list = []

    for i, player_ids in enumerate(flat1_df_stats['player_ids']):
        mapping = {}
        for player_id, player_info in player_ids.items():
            # If the role is 'self', map it to 'ally'
            role = 'ally' if player_info['role'] == 'self' else player_info['role']
            column_name = f"{role}{player_id}_character"
            mapping[column_name] = player_info['character']
        
        # If mapping is not empty
        if mapping:
            expanded_player_ids_list.append(mapping)
            
    # Create the expanded_player_ids DataFrame
    expanded_player_ids = pd.DataFrame(expanded_player_ids_list)

    # Concatenate the original DataFrame with the expanded columns
    flat2_df_stats_chars = pd.concat([flat1_df_stats.drop('player_ids', axis=1), expanded_player_ids], axis=1)

    ### Flattening the round_info/round_totals cols by 1 level
    # This is where the DF expands into rounds being the rows instead of matches being the rows
    # Have to parse the round info/round totals at the same time
    
    # Initialize an empty dictionary to store map names and unique areas
    map_areas = {}
    for index, row in flat2_df_stats_chars.iterrows():
        # for all the rounds in each match
        for round_number, round_data in row['round_info'].items():
            # get the current/original row.
            new_row = row.to_dict()
            # exclude the original complex columns that we're flattening
            new_row.pop('rounds', None)
            new_row.pop('round_info', None)
            
            # get current map
            cur_map = new_row['map']

            # Initialize a new set for this map if it doesn't exist in map_areas
            if cur_map not in map_areas:
                map_areas[cur_map] = set()
                
            # Add round number to the new row
            new_row['round_number'] = round_number
            

            # Flatten rounds data into the new row
            for key, value in row['rounds'][round_number].items():
                if key == 'map_region':
                    # value should be a list of lists where each sublist's second element is an area
                    for sublist in value:
                        # Add the area to the set corresponding to the current map
                        map_areas[cur_map].add(sublist[1])

    ### This function handles most of the "pre/post" spike split metrics ###
    # needs to consider max, total, and avg
    def calculate_metrics(key, value, cur_player, new_row, spike_time):
        pre_spike_total = 0
        post_spike_total = 0
        pre_spike_entries = 0
        post_spike_entries = 0
        pre_spike_max_loss = 0
        post_spike_max_loss = 0
        pre_spike_total_loss = 0
        post_spike_total_loss = 0
        last_value = None

        for entry in value:
            entry_time = entry[0]
            entry_value = entry[1]

            if last_value is not None:
                value_loss = max(0, last_value - entry_value)
                if spike_time is None or entry_time < spike_time:
                    pre_spike_max_loss = max(pre_spike_max_loss, value_loss)
                    pre_spike_total_loss += value_loss
                elif spike_time:
                    post_spike_max_loss = max(post_spike_max_loss, value_loss)
                    post_spike_total_loss += value_loss

            last_value = entry_value

            if spike_time is None or entry_time < spike_time:
                pre_spike_total += entry_value
                pre_spike_entries += 1
            elif spike_time:
                post_spike_total += entry_value
                post_spike_entries += 1

        pre_spike_avg = pre_spike_total / pre_spike_entries if pre_spike_entries > 0 else 0
        post_spike_avg = post_spike_total / post_spike_entries if post_spike_entries > 0 else 0

        new_row[f'ally{cur_player}_pre_spike_avg_{key}'] = pre_spike_avg
        new_row[f'ally{cur_player}_post_spike_avg_{key}'] = post_spike_avg
        new_row[f'ally{cur_player}_pre_spike_max_{key}_loss'] = pre_spike_max_loss
        new_row[f'ally{cur_player}_post_spike_max_{key}_loss'] = post_spike_max_loss
        new_row[f'ally{cur_player}_pre_spike_total_{key}_loss'] = pre_spike_total_loss
        new_row[f'ally{cur_player}_post_spike_total_{key}_loss'] = post_spike_total_loss

    def calculate_longest_duration(key, value, cur_player, new_row, spike_time):
        current_item = None
        longest_duration = 0
        longest_item = None
        pre_spike_longest_item = None
        post_spike_longest_item = None
        pre_spike_longest_duration = 0
        post_spike_longest_duration = 0

        for entry in value:
            timestamp = entry[0]
            item = entry[1]

            if current_item is None:
                current_item = item
                start_time = timestamp
            elif item != current_item:
                duration = timestamp - start_time
                if duration > longest_duration:
                    longest_duration = duration
                    longest_item = current_item

                if spike_time is None or start_time < spike_time:  # pre-spike or no spike
                    if duration > pre_spike_longest_duration:
                        pre_spike_longest_duration = duration
                        pre_spike_longest_item = current_item
                elif spike_time and start_time >= spike_time:  # post-spike
                    if duration > post_spike_longest_duration:
                        post_spike_longest_duration = duration
                        post_spike_longest_item = current_item

                current_item = item
                start_time = timestamp

        new_row[f'ally{cur_player}_longest_{key}'] = longest_item
        new_row[f'ally{cur_player}_pre_spike_longest_{key}'] = pre_spike_longest_item
        new_row[f'ally{cur_player}_post_spike_longest_{key}'] = post_spike_longest_item

    ### Since we have charges I need to track when a player has a charge, and that charge goes down ###
    def calculate_ability_usage(cur_player, ability_charges, spike_time, identifier, new_row):
        pre_spike_total_ability_usage = 0
        post_spike_total_ability_usage = 0

        for i in range(len(ability_charges) - 1):
            ability_time = ability_charges[i + 1][0]
            current_charge = ability_charges[i + 1][1]
            previous_charge = ability_charges[i][1]

            if previous_charge > current_charge:
                if spike_time is None or ability_time < spike_time:  # pre-spike or no spike
                    pre_spike_total_ability_usage += previous_charge - current_charge
                elif spike_time:  # post-spike
                    post_spike_total_ability_usage += previous_charge - current_charge

        new_row[f'ally{cur_player}_pre_spike_total_ability_usage_{identifier}'] = pre_spike_total_ability_usage
        new_row[f'ally{cur_player}_post_spike_total_ability_usage_{identifier}'] = post_spike_total_ability_usage

    # First pass: get spike plant time for each round
    spike_times = {}  # Create a dictionary to store the spike_time for each round
    for index, row in flat2_df_stats_chars.iterrows():
        for round_number, round_data in row['round_info'].items():
            for key, value in row['rounds'][round_number].items():
                if key == 'spike_planted':
                    # Initialize default values
                    spike_yn = False
                    time = 0
                    rounds_spike_planted = value
                    # Iterate over the list to find the first instance of spike planted
                    for item in value:
                        if item[1]:  # Check if spike was planted
                            spike_yn = item[1]
                            time = item[0]
                            break  # Stop iterating after finding the first instance
    
                    spike_times[(index, round_number)] = time if spike_yn else None
    
    new_rows = []

    # Second pass: compute metrics
    # for index, row in flat2_df_stats_chars.iterrows():
    for index, row in flat2_df_stats_chars.iterrows():
        for round_number, round_data in row['round_info'].items():
            # get the current/original row.
            new_row = row.to_dict()
            # Get spike time for this round
            spike_time = spike_times[(index, round_number)]
            if spike_time:
                new_row['spike_planted'] = True
                new_row['spike_time'] = spike_time
            else:
                new_row['spike_planted'] = False
                new_row['spike_time'] = 0
            # get current map
            cur_map = new_row['map']
            # get current player
            cur_player = new_row['player']
            # Add round number to the new row
            new_row['round_number'] = round_number
            # Flatten round_info data into the new row
            for key, value in round_data.items():
                new_row[f'round_info_{key}'] = value
            # get side
            side = new_row['round_info_ally_side']
            # get won 
            # won = new_row['round_info_round_won']


            for key, value in row['rounds'][round_number].items():
                ## COUNTING ELIMINATIONS ##
                if key == 'elims':

                    rounds_elims = value

                    # Initialize the stats for all players
                    player_stats = {str(player): {
                    "pre_spike_elims": 0, "post_spike_elims": 0,
                    "pre_spike_deaths": 0, "post_spike_deaths": 0,
                    "pre_spike_assists": 0, "post_spike_assists": 0,
                    "pre_spike_headshots": 0, "post_spike_headshots": 0,
                    "pre_spike_wallbangs": 0, "post_spike_wallbangs": 0,
                    "pre_spike_first_bloods": 0, "post_spike_first_bloods": 0} for player in range(10)}

                    ally_deaths = 0
                    opponent_deaths = 0

                    # Then, for each elimination:
                    for elimination in rounds_elims:
                        elim_data = elimination[1]
                        source = elim_data['source']
                        target = elim_data['target']
                        assisted = elim_data['assisted']

                        # check if the elimination happened before or after the spike
                        if spike_time is None or elimination[0] < spike_time:  # pre-spike
                            player_stats[source]['pre_spike_elims'] += 1
                            player_stats[target]['pre_spike_deaths'] += 1
                            for assist_player in assisted:
                                player_stats[assist_player]['pre_spike_assists'] += 1
                            if elim_data['headshot']:
                                player_stats[source]['pre_spike_headshots'] += 1
                            if elim_data['wallbang']:
                                player_stats[source]['pre_spike_wallbangs'] += 1
                            if elim_data['first_blood']:
                                player_stats[source]['pre_spike_first_bloods'] += 1
                        elif spike_time:  # post-spike, only if spike_time is not None
                            player_stats[source]['post_spike_elims'] += 1
                            player_stats[target]['post_spike_deaths'] += 1
                            for assist_player in assisted:
                                player_stats[assist_player]['post_spike_assists'] += 1
                            if elim_data['headshot']:
                                player_stats[source]['post_spike_headshots'] += 1
                            if elim_data['wallbang']:
                                player_stats[source]['post_spike_wallbangs'] += 1
                            if elim_data['first_blood']:
                                player_stats[source]['post_spike_first_bloods'] += 1

                        if int(target) < 5:  # If the target is an ally
                            ally_deaths += 1
                        else:  # If the target is an opponent
                            opponent_deaths += 1

                    all_ally_dead = ally_deaths >= 5
                    all_opponent_dead = opponent_deaths >= 5

                    # Add these stats to the new_row, outside the eliminations loop:
                    for player, stats in player_stats.items():
                        prefix = 'ally' if int(player) < 5 else 'opponent'
                        player_num = player if int(player) < 5 else str(int(player) - 5)
                        for stat, count in stats.items():
                            new_row[f'{prefix}{player_num}_{stat}'] = count

                    new_row['all_ally_dead'] = all_ally_dead
                    new_row['all_opponent_dead'] = all_opponent_dead

                ## CALC AVG HEALTH AND MAX/TOTAL HEALTH LOSS ##
                if key == 'health':
                    rounds_health = value
                    for player, health_data in rounds_health.items():
                        pre_spike_total_health = 0
                        post_spike_total_health = 0
                        pre_spike_entries = 0
                        post_spike_entries = 0
                        pre_spike_max_health_loss = 0
                        post_spike_max_health_loss = 0
                        pre_spike_total_health_loss = 0
                        post_spike_total_health_loss = 0
                        last_health = None
                        for entry in health_data:
                            health_time = entry[0]
                            health_value = entry[1]

                            if last_health is not None:  # if there is a previous health value, calculate the health loss
                                health_loss = max(0, last_health - health_value)  # don't let health loss be negative
                                if spike_time is None or health_time < spike_time:  # pre-spike or no spike
                                    pre_spike_max_health_loss = max(pre_spike_max_health_loss, health_loss)
                                    pre_spike_total_health_loss += health_loss
                                elif spike_time:  # post-spike
                                    post_spike_max_health_loss = max(post_spike_max_health_loss, health_loss)
                                    post_spike_total_health_loss += health_loss

                            last_health = health_value  # store the current health value as the last health value for the next iteration

                            if spike_time is None or health_time < spike_time:  # pre-spike or no spike
                                pre_spike_total_health += health_value
                                pre_spike_entries += 1
                            elif spike_time:  # post-spike
                                post_spike_total_health += health_value
                                post_spike_entries += 1

                        pre_spike_average_health = pre_spike_total_health / pre_spike_entries if pre_spike_entries > 0 else 0
                        post_spike_average_health = post_spike_total_health / post_spike_entries if post_spike_entries > 0 else 0

                        new_row[f'ally{player}_pre_spike_avg_health'] = pre_spike_average_health
                        new_row[f'ally{player}_post_spike_avg_health'] = post_spike_average_health
                        new_row[f'ally{player}_pre_spike_max_health_loss'] = pre_spike_max_health_loss
                        new_row[f'ally{player}_post_spike_max_health_loss'] = post_spike_max_health_loss
                        new_row[f'ally{player}_pre_spike_total_health_loss'] = pre_spike_total_health_loss
                        new_row[f'ally{player}_post_spike_total_health_loss'] = post_spike_total_health_loss
                        
                
                ## CALC AVG SHIELD AND MAX/TOTAL SHIELD LOSS ##
                if key == 'shield':
                    calculate_metrics(key, value, cur_player, new_row, spike_time)
                ## CALC AVG CREDITS AND MAX/TOTAL CREDIT LOSS ##
                if key == 'credits':
                    calculate_metrics(key, value, cur_player, new_row, spike_time)
                ## CALC AVG AMMO MAG AND MAX/TOTAL AMMO MAG LOSS ##  
                if key == 'ammo_mag':
                    calculate_metrics(key, value, cur_player, new_row, spike_time)
                ## MOST TIME/MAX INV STATE ##
                if key == 'inv_state':
                    calculate_longest_duration('inv_state', value, cur_player, new_row, spike_time)
                
                ## COUNTING ULTIMATE USAGE ##
                if key == 'ult_state':
                    rounds_ult_state = value

                    for player, ultimate_data in rounds_ult_state.items():
                        ultimate_usage = 0
                        pre_spike_ultimate_usage = 0
                        post_spike_ultimate_usage = 0
                        previous_state = None
                        for entry in ultimate_data:
                            ultimate_state_time = entry[0]
                            ultimate_state = entry[1]
                            if previous_state is not None and previous_state is True and ultimate_state is False:
                                ultimate_usage += 1
                                if spike_time is None or ultimate_state_time < spike_time:  # pre-spike or no spike
                                    pre_spike_ultimate_usage += 1
                                elif spike_time and ultimate_state_time >= spike_time:  # post-spike
                                    post_spike_ultimate_usage += 1
                            previous_state = ultimate_state
                        if int(player) < 5:
                            new_row[f'ally{player}_ultimate_usage'] = ultimate_usage
                            new_row[f'ally{player}_pre_spike_ultimate_usage'] = pre_spike_ultimate_usage
                            new_row[f'ally{player}_post_spike_ultimate_usage'] = post_spike_ultimate_usage
                        else:
                            new_row[f'opponent{int(player)-5}_ultimate_usage'] = ultimate_usage
                            new_row[f'opponent{int(player)-5}_pre_spike_ultimate_usage'] = pre_spike_ultimate_usage
                            new_row[f'opponent{int(player)-5}_post_spike_ultimate_usage'] = post_spike_ultimate_usage

                ## GETTING % MAP COVERED AND MOVEMENT METRIC"
                if key == 'map_region':
                    rounds_map_region = value

                    # % map covered
                    available_places = map_areas[cur_map]
                    visited_places_pre_spike = set()
                    visited_places_post_spike = set()

                    transitions_pre_spike = set()
                    transitions_post_spike = set()

                    previous_place_pre_spike = None
                    previous_place_post_spike = None

                    for entry in rounds_map_region:
                        timestamp = entry[0]
                        place = entry[1]

                        if spike_time is None or timestamp < spike_time:  # Pre spike or no spike planted
                            if place in available_places:
                                visited_places_pre_spike.add(place)

                            if previous_place_pre_spike is not None:
                                transition = (previous_place_pre_spike, place)
                                transitions_pre_spike.add(transition)

                            previous_place_pre_spike = place
                        else:  # Post spike
                            if place in available_places:
                                visited_places_post_spike.add(place)

                            if previous_place_post_spike is not None:
                                transition = (previous_place_post_spike, place)
                                transitions_post_spike.add(transition)

                            previous_place_post_spike = place

                    metric_pre_spike = len(visited_places_pre_spike) / len(available_places)
                    new_row[f'ally{cur_player}_pre_spike_map_covered'] = metric_pre_spike
                    movement_metric_pre_spike = len(transitions_pre_spike)
                    new_row[f'ally{cur_player}_pre_spike_movement_metric'] = movement_metric_pre_spike

                    metric_post_spike = len(visited_places_post_spike) / len(available_places)
                    new_row[f'ally{cur_player}_post_spike_map_covered'] = metric_post_spike
                    movement_metric_post_spike = len(transitions_post_spike)
                    new_row[f'ally{cur_player}_post_spike_movement_metric'] = movement_metric_post_spike

                
                ## MOST TIME/MAX PRIMARY GUN ##
                if key == 'inv_primary':
                    calculate_longest_duration('gun_primary', value, cur_player, new_row, spike_time)
                ## CALC AVG AMMO RESERVE AND MAX/TOTAL AMMO RESERVE LOSS ##  
                if key == 'ammo_reserve':
                    calculate_metrics(key, value, cur_player, new_row, spike_time)
                
                ## TOTAL FIRING TIME ##
                if key == 'firing_state':
                    firing_state = value

                    total_fire_time = 0
                    pre_spike_total_fire_time = 0
                    post_spike_total_fire_time = 0
                    start_fire = 0
                    firing = 0

                    for i in range(len(firing_state) - 1):
                        if firing_state[i + 1][1] == True:
                            start_fire = firing_state[i + 1][0]
                        elif firing_state[i + 1][1] == False:
                            end_fire = firing_state[i + 1][0]
                            firing = end_fire - start_fire
                            total_fire_time += firing

                            if spike_time is None or start_fire < spike_time:  # pre-spike or no spike
                                pre_spike_total_fire_time += firing
                            elif spike_time and start_fire >= spike_time:  # post-spike
                                post_spike_total_fire_time += firing

                    new_row[f'ally{cur_player}_total_firing_time'] = total_fire_time
                    new_row[f'ally{cur_player}_pre_spike_total_firing_time'] = pre_spike_total_fire_time
                    new_row[f'ally{cur_player}_post_spike_total_firing_time'] = post_spike_total_fire_time
                else:
                    new_row[f'rounds_{key}'] = value   
                
                ## MOST TIME/MAX SECONDARY GUN ##
                if key == 'inv_secondary':
                    calculate_longest_duration('gun_secondary', value, cur_player, new_row, spike_time)
                ## CALC AVG LOADOUT VALUE AND MAX/TOTAL LOADOUT VALUE LOSS ##
                if key == 'loadout_value':
                    calculate_metrics(key, value, cur_player, new_row, spike_time)
                
                ## CALC ABILITY USE ##
                if key in ['ability_charges_1', 'ability_charges_2', 'ability_charges_3', 'ability_charges_4']:
                    ability_charges = value
                    identifier = key.split('_')[-1]  # extract the ability identifier from the key
                    calculate_ability_usage(cur_player, ability_charges, spike_time, identifier, new_row)

            # Getting rid of cols that don't need 
            new_row.pop('rounds_alive', None)
            new_row.pop('rounds_phases', None)
            new_row.pop('rounds_spike_planted', None)
            new_row.pop('rounds_assisted_count', None)
            new_row.pop('rounds_assists_count', None)
            new_row.pop('rounds_elims_count', None)
            new_row.pop('round_totals', None)
            new_row.pop('round_info_ally_score', None)
            new_row.pop('round_info_opponent_score', None)

            # Getting rid of cols that we just parsed
            new_row.pop('rounds_elims', None)
            new_row.pop('rounds_health', None)
            new_row.pop('rounds_shield', None)
            new_row.pop('rounds_credits', None)
            new_row.pop('rounds_ammo_mag', None)
            new_row.pop('rounds_inv_state', None)
            new_row.pop('rounds_ult_state', None)
            new_row.pop('rounds_map_region', None)
            new_row.pop('rounds_inv_primary', None)
            new_row.pop('rounds_ammo_reserve', None)
            new_row.pop('rounds_firing_state', None)
            new_row.pop('rounds_inv_secondary', None)
            new_row.pop('rounds_loadout_value', None)
            new_row.pop('rounds_ability_charges_1', None)
            new_row.pop('rounds_ability_charges_2', None)
            new_row.pop('rounds_ability_charges_3', None)
            new_row.pop('rounds_ability_charges_4', None)
            
            # Getting rid of cols that we don't want/need    
            new_row.pop('round_info_score', None)
            new_row.pop('round_info_round_scored', None)
            new_row.pop('round_info_buy_start', None)
            new_row.pop('round_info_ult_used', None)
            new_row.pop('round_info_spike_planted', None)

            # exclude the original complex columns that we're flattening
            new_row.pop('rounds', None)
            new_row.pop('round_info', None)

            # fixing the naming of the players
            def adjust_keys(new_row, cur_player):
                adjusted_row = {}
                cur_player = int(cur_player)

                allies_count = 5  # define total number of allies, including 'self'

                for key, value in new_row.items():
                    if key.startswith(f'ally{cur_player}'):
                        adjusted_key = key.replace(f'ally{cur_player}', 'self')
                    elif key.startswith('ally'):
                        # correctly parse the ally number as an integer and remaining part of the key
                        ally_number = int(key[4:].split('_')[0])
                        ally_remaining = '_'.join(key.split('_')[1:])

                        # calculate new ally number
                        new_ally_number = (ally_number - cur_player) % allies_count

                        # construct new key with updated ally number
                        adjusted_key = 'ally' + str(new_ally_number) + '_' + ally_remaining
                    else:
                        adjusted_key = key

                    adjusted_row[adjusted_key] = value

                return adjusted_row

            new_row = adjust_keys(new_row, cur_player)

            # Append this new row to the list
            new_rows.append(new_row)

    # Convert the list of new rows into a dataframe
    new_df = pd.DataFrame(new_rows)

    ### Data cleaning
    # I noticed some NaNs. This happens when there is missing information
    # self_longest_gun_primary, self_longest_inv_state, self_longest_gun_secondaryis coming up as None but i think it should be 'none', so i am replacing with that
    # After investigating it seems like the elims data isnt always captured, its only 123 rows so im going to drop it

    new_df['self_pre_spike_longest_gun_primary'] = new_df['self_pre_spike_longest_gun_primary'].fillna('none')
    new_df['self_post_spike_longest_gun_primary'] = new_df['self_post_spike_longest_gun_primary'].fillna('none')
    new_df['self_pre_spike_longest_gun_secondary'] = new_df['self_pre_spike_longest_gun_secondary'].fillna('none')
    new_df['self_post_spike_longest_gun_secondary'] = new_df['self_post_spike_longest_gun_secondary'].fillna('none')
    new_df['self_pre_spike_longest_inv_state'] = new_df['self_pre_spike_longest_inv_state'].fillna('none')
    new_df['self_post_spike_longest_inv_state'] = new_df['self_post_spike_longest_inv_state'].fillna('none')
    new_df['self_longest_inv_state'] = new_df['self_longest_inv_state'].fillna('none')
    new_df['self_longest_gun_primary'] = new_df['self_longest_gun_primary'].fillna('none')
    new_df['self_longest_gun_secondary'] = new_df['self_longest_gun_secondary'].fillna('none')

    clean_df = new_df.dropna()
    clean_df = clean_df.reset_index(drop=True)  

    clean_df.to_csv(csv_title, index=False)

### CREATING THE CSVs ###
create_norm_csv_spike('parsing/dev_data.json', 'dev_data_norm_round_spikesplit.csv')
create_norm_csv_spike('parsing/pro1_data.json', 'pro1_data_norm_round_spikesplit.csv')
create_norm_csv_spike('parsing/pro2_data.json', 'pro2_data_norm_round_spikesplit.csv')
