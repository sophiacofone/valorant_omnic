# 7/4/23, Sophia Cofone, Omnic ML Project
# Purpose of this script is to parse & normalize the data by round (TOTALS)
# Works great with dev_data, pro1_data, and pro2_data
# Code is pretty manual/hardcoded. If the strucutre of the data changes or features are added, the code will need to be updated/reviewed.

# required imports
import json
import pandas as pd

def create_norm_csv_total(json_filename,csv_title):
    # loading the data
    with open(json_filename, 'r') as f:
        data = json.load(f)

    ### Create df ###
    df = pd.read_json(json_filename)
    # Drop the all cols besides 'statistics' and 'user_id'
    df_stats = df[['user_id','statistics']]

    ### Flattening the stats col by 1 level ###
    flat1_df_stats = pd.json_normalize(df_stats['statistics'], max_level=0)
    flat1_df_stats = pd.concat([df_stats['user_id'], flat1_df_stats], axis=1)

    ### Removing some cols we don't care about from the ROUND perspective ###
    # Keep in mind the 'totals', 'result' are from the MATCH perspective
    # even though the map is from match perspective I need it so I am adding it in (similar to userID)
    # # weapon stuff is all from match perspective, so I am dropping (some exists in rounds data anyway)
    # I think its possible to do something with 'allies_onscreen', 'opponents_onscreen', and 'detections_totals' but I am dropping for now
    flat1_df_stats = flat1_df_stats.drop(['totals','score', 'allies','result', 'status','gametype','version','end_time','opponents','processed','ally_score','start_time','opponent_score','detections_totals','best_weapon_elims','most_used_primary_seconds','analysis_processed','most_used_secondary_seconds','best_weapon_type_elims','opponents_onscreen','allies_onscreen','analysis_status','player_totals','best_weapon_type','best_weapon','most_used_secondary','most_used_primary','best_weapon_type_elims'], axis=1)

    ### Flattening the player_ids col by 1 level ###
    # this is getting all the teammates and opponents, as well as mapping the 'player' to an ally
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

    ### Flattening the round_info/round_totals cols by 1 level ###
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
    new_rows = []
    # for all the matches in flat2_df_stats_chars...
    for index, row in flat2_df_stats_chars.iterrows():
        # for all the rounds in each match
        for round_number, round_data in row['round_info'].items():
            # get the current/original row.
            new_row = row.to_dict()
            
            # get current map
            cur_map = new_row['map']
            
            # get current player
            cur_player = new_row['player']
            
            # Add round number to the new row
            new_row['round_number'] = round_number
            
            # Flatten round_info data into the new row
            for key, value in round_data.items():
                new_row[f'round_info_{key}'] = value
            

            # Flatten rounds data into the new row
            for key, value in row['rounds'][round_number].items():
                
                ## UNIQUE ##
                if key == 'spike_planted':
                    rounds_spike_planted = value

                    # Initialize default values
                    spike_yn = False
                    time = value[0][0]

                    # Iterate over the list to find the first instance of spike planted
                    for item in value:
                        if item[1]:  # Check if spike was planted
                            spike_yn = item[1]
                            time = item[0]
                            break  # Stop iterating after finding the first instance

                    new_row['spike_planted'] = spike_yn
                    new_row['spike_time'] = time
                
                if key == 'map_region':
                    rounds_map_region = value
                    
                    # % map covered
                    avalible_places = map_areas[cur_map]
                    visited_places = set()

                    for entry in rounds_map_region:
                        place = entry[1]
                        if place in avalible_places:
                            visited_places.add(place)

                    metric = len(visited_places) / len(avalible_places)
                    
                    new_row[f'ally{cur_player}_prec_map_covered'] = metric
                    
                    # 'movement' metric (based on transisions between areas)
                    transitions = set()
                    previous_place = None
                    movement_metric = 0

                    for entry in rounds_map_region:
                        place = entry[1]
                        if previous_place is not None:
                            transition = (previous_place, place)
                            transitions.add(transition)
                        previous_place = place

                    movement_metric = len(transitions)
                    f'ally{cur_player}_movement_metric'
                    new_row[f'ally{cur_player}_movement_metric'] = movement_metric
                
                ## MOST TIME/MAX ##
                if key == 'inv_primary':
                    rounds_inv_primary = value
                    
                    current_gun = None
                    longest_duration = 0
                    start_time = rounds_inv_primary[0][0]
                    end_time = rounds_inv_primary[-1][0]

                    longest_gun = None

                    for entry in rounds_inv_primary:
                        timestamp = entry[0]
                        gun = entry[1]

                        if current_gun is None:
                            current_gun = gun
                        elif gun != current_gun:
                            duration = timestamp - start_time
                            if duration > longest_duration:
                                longest_duration = duration
                                longest_gun = current_gun
                                gun_time = (longest_gun, longest_duration)

                            current_gun = gun
                            start_time = timestamp
                    
                    new_row[f'ally{cur_player}_longest_gun_primary'] = longest_gun
                    
                if key == 'inv_secondary':
                    rounds_inv_secondary = value
                    
                    current_gun = None
                    longest_duration = 0
                    start_time = rounds_inv_secondary[0][0]
                    end_time = rounds_inv_secondary[-1][0]

                    longest_gun = None

                    for entry in rounds_inv_secondary:
                        timestamp = entry[0]
                        gun = entry[1]

                        if current_gun is None:
                            current_gun = gun
                        elif gun != current_gun:
                            duration = timestamp - start_time
                            if duration > longest_duration:
                                longest_duration = duration
                                longest_gun = current_gun
                                gun_time = (longest_gun, longest_duration)

                            current_gun = gun
                            start_time = timestamp
                    
                    new_row[f'ally{cur_player}_longest_gun_secondary'] = longest_gun
                    
                if key == 'inv_state':
                    rounds_inv_state = value

                    current_inv_state = None
                    longest_duration = 0
                    longest_inv_state = None

                    if rounds_inv_state:  # Check if the list is not empty
                        start_time = rounds_inv_state[0][0]
                        end_time = rounds_inv_state[-1][0]

                        for entry in rounds_inv_state:
                            timestamp = entry[0]
                            inv_state = entry[1]

                            if current_inv_state is None:
                                current_inv_state = inv_state
                            elif inv_state != current_inv_state:
                                duration = timestamp - start_time
                                if duration > longest_duration:
                                    longest_duration = duration
                                    longest_inv_state = current_inv_state

                                current_inv_state = inv_state
                                start_time = timestamp

                    new_row[f'ally{cur_player}_longest_inv_state'] = longest_inv_state

                
                ## COUNT ##
                if key == 'ult_state':
                    rounds_ult_state = value
                    
                    for player, ultimate_data in rounds_ult_state.items():
                        ultimate_usage = 0
                        previous_state = None
                        for entry in ultimate_data:
                            ultimate_state = entry[1]
                            if previous_state is not None and previous_state is True and ultimate_state is False:
                                ultimate_usage += 1
                            previous_state = ultimate_state
                        if int(player) < 5:
                            new_row[f'ally{player}_ultimate_usage'] = ultimate_usage
                        else:
                            new_row[f'opponent{int(player)-5}_ultimate_usage'] = ultimate_usage
                
                ## COUNT ##
                if key == 'elims':
                    rounds_elims = value
                    # Initialize the stats for all players
                    player_stats = {
                        str(player): {"elims": 0, "deaths": 0, "assists": 0, "headshots": 0, "wallbangs": 0, "first_bloods": 0} 
                        for player in range(10)}
                    
                    # Then, for each elimination:
                    for elimination in rounds_elims:
                        elim_data = elimination[1]
                        source = elim_data['source']
                        target = elim_data['target']
                        assisted = elim_data['assisted']

                        # Increase elim count for the source player
                        player_stats[source]['elims'] += 1

                        # Increase death count for the target player
                        player_stats[target]['deaths'] += 1

                        # Increase assist count for all assisted players
                        for assist_player in assisted:
                            player_stats[assist_player]['assists'] += 1

                        # If the elimination was a headshot, increase headshot count for the source player
                        if elim_data['headshot']:
                            player_stats[source]['headshots'] += 1

                        # If the elimination was a wallbang, increase wallbang count for the source player
                        if elim_data['wallbang']:
                            player_stats[source]['wallbangs'] += 1

                        # If the elimination was the first blood, increase first_blood count for the source player
                        if elim_data['first_blood']:
                            player_stats[source]['first_bloods'] += 1

                        # Add these stats to the new_row:
                        for player, stats in player_stats.items():
                            prefix = 'ally' if int(player) < 5 else 'opponent'
                            player_num = player if int(player) < 5 else str(int(player) - 5)
                            for stat, count in stats.items():
                                new_row[f'{prefix}{player_num}_{stat}'] = count
                
                ## AVERAGES ## 
                if key == 'ammo_reserve':
                    rounds_ammo_reserve = value
                    sum_ammo_reserve = 0
                    num_entries = len(rounds_ammo_reserve)

                    if num_entries > 0:
                        for entry in rounds_ammo_reserve:
                            ammo_reserve_value = entry[1]
                            sum_ammo_reserve += ammo_reserve_value

                        average_ammo_reserve = sum_ammo_reserve / num_entries

                    else:
                        average_ammo_reserve = 0
                        
                
                    new_row[f'ally{cur_player}_avg_ammo_reserve'] = average_ammo_mag
                    
                if key == 'ammo_mag':
                    rounds_ammo_mag = value
                    sum_ammo_mag = 0
                    num_entries = len(rounds_ammo_mag)
                    
                    if num_entries > 0:
                        for entry in rounds_ammo_mag:
                            ammo_ammo_mag = entry[1]
                            sum_ammo_mag += ammo_ammo_mag

                        average_ammo_mag = sum_ammo_mag/ num_entries
                    else:
                        average_ammo_mag = 0
                    
                    new_row[f'ally{cur_player}_avg_ammo_mag'] = average_ammo_mag
                    
                if key == 'credits':
                    rounds_credits = value
                    sum_credits = 0
                    num_entries = len(rounds_credits)
                    
                    if num_entries > 0:
                        for entry in rounds_credits:
                            credits_value = entry[1]
                            sum_credits += credits_value

                        average_credits = sum_credits / num_entries
                    else:
                        average_credits = 0
                    
                    new_row[f'ally{cur_player}_avg_credits'] = average_credits
                    
                if key == 'shield':
                    rounds_shield = value
                    sum_shield = 0
                    num_entries = len(rounds_shield)
                    
                    if num_entries > 0:
                        for entry in rounds_shield:
                            shield_value = entry[1]
                            sum_shield += shield_value

                        average_shield = sum_shield / num_entries
                    else:
                        average_shield = 0
                    
                    new_row[f'ally{cur_player}_avg_shield'] = average_shield
                
                if key == 'health':
                    rounds_health = value
                    for player, health_data in rounds_health.items():
                        total_health = 0
                        num_entries = len(health_data)
                        if num_entries > 0:
                            for entry in health_data:
                                health_value = entry[1]
                                total_health += health_value
                            average_health = total_health / num_entries
                        else:
                            average_health = 0
                        new_row[f'ally{player}_avg_health'] = average_health
                
                ## MAXS ## 
                if key == 'loadout_value':
                    rounds_loadout_value = value
                    max_loadout_value = max(rounds_loadout_value, key=lambda x: x[1])
                    max_loadout_value = max_loadout_value[1]
                    
                    new_row[f'ally{cur_player}_max_loadout_value'] = max_loadout_value
                
                ## TOTALS ##         
                if key == 'ability_charges_1':
                    rounds_ability_charges_1 = value
                    total_ability_usage_1 = 0

                    for i in range(len(rounds_ability_charges_1) - 1):
                        if rounds_ability_charges_1[i + 1][1] == 1:
                            total_ability_usage_1 += 1
                    
                    new_row[f'ally{cur_player}_total_ability_usage_1'] = total_ability_usage_1
                    
                if key == 'ability_charges_2':
                    rounds_ability_charges_2 = value
                    total_ability_usage_2 = 0

                    for i in range(len(rounds_ability_charges_2) - 1):
                        if rounds_ability_charges_2[i + 1][1] == 1:
                            total_ability_usage_2 += 1
                    
                    new_row[f'ally{cur_player}_total_ability_usage_2'] = total_ability_usage_2
                    
                if key == 'ability_charges_3':
                    rounds_ability_charges_3 = value
                    total_ability_usage_3 = 0

                    for i in range(len(rounds_ability_charges_3) - 1):
                        if rounds_ability_charges_3[i + 1][1] == 1:
                            total_ability_usage_3 += 1
                    
                    new_row[f'ally{cur_player}_total_ability_usage_3'] = total_ability_usage_3
                    
                if key == 'ability_charges_4':
                    rounds_ability_charges_4 = value
                    total_ability_usage_4 = 0

                    for i in range(len(rounds_ability_charges_4) - 1):
                        if rounds_ability_charges_4[i + 1][1] == 1:
                            total_ability_usage_4 += 1
                    
                    new_row[f'ally{cur_player}_total_ability_usage_4'] = total_ability_usage_4
                    
                if key == 'firing_state':
                    firing_state = value

                    total_fire_time = 0
                    start_fire = 0

                    for i in range(len(firing_state) - 1):
                        if firing_state[i + 1][1] == True:
                            start_fire = firing_state[i + 1][0]
                        elif firing_state[i + 1][1] == False:
                            end_fire = firing_state[i + 1][0]
                            firing = end_fire - start_fire
                            total_fire_time += firing
                    
                    new_row[f'ally{cur_player}_total_firing_time'] = total_fire_time
                else:
                    new_row[f'rounds_{key}'] = value

            # Getting rid of cols that we just parsed or dont need 
            new_row.pop('rounds_firing_state', None)
            new_row.pop('rounds_ability_charges_1', None)
            new_row.pop('rounds_ability_charges_2', None)
            new_row.pop('rounds_ability_charges_3', None)
            new_row.pop('rounds_ability_charges_4', None)
            new_row.pop('rounds_assisted_count', None)
            new_row.pop('rounds_assists_count', None)
            new_row.pop('rounds_loadout_value', None)
            new_row.pop('rounds_ammo_reserve', None)
            new_row.pop('rounds_map_region', None)
            new_row.pop('rounds_phases', None)
            new_row.pop('rounds_ammo_mag', None)
            new_row.pop('rounds_credits', None)
            new_row.pop('rounds_shield', None)
            new_row.pop('rounds_credits', None)
            new_row.pop('rounds_inv_primary', None)
            new_row.pop('rounds_inv_secondary', None)
            new_row.pop('rounds_inv_state', None)
            new_row.pop('rounds_health', None)
            new_row.pop('rounds_ult_state', None)
            new_row.pop('round_totals', None)
            new_row.pop('rounds_alive', None)
            new_row.pop('round_info_ally_score', None)
            new_row.pop('round_info_opponent_score', None)
            new_row.pop('rounds_elims', None)
            new_row.pop('rounds_elims_count', None)
            new_row.pop('rounds_spike_planted', None)
            # exclude the original complex columns that we're flattening
            new_row.pop('rounds', None)
            new_row.pop('round_info', None)
            # Getting rid of cols that we dont want/need    
            new_row.pop('round_info_score', None)
            new_row.pop('round_info_round_scored', None)
            new_row.pop('round_info_buy_start', None)
            new_row.pop('round_info_ult_used', None)
            new_row.pop('round_info_spike_planted', None)
        
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

    ### Data cleaning ###
    # I noticed some NaNs. This happens when there is missing information
    # self_longest_gun_primary, self_longest_inv_state, self_longest_gun_secondaryis coming up as None but i think it should be 'none', so i am replacing with that
    # After investigating it seems like the elims data isnt always captured, its only 123 rows so im going to drop it
    new_df['self_longest_gun_primary'] = new_df['self_longest_gun_primary'].fillna('none')
    new_df['self_longest_gun_secondary'] = new_df['self_longest_gun_secondary'].fillna('none')
    new_df['self_longest_inv_state'] = new_df['self_longest_inv_state'].fillna('none')

    clean_df = new_df.dropna()
    clean_df = clean_df.reset_index(drop=True)

    clean_df.to_csv(csv_title, index=False)





