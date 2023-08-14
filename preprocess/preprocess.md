# Preprocessing
This project required extensive data preperation. Some of the preprocessing is consistant across all the models, and other preprocessing is unique to that model.

## Preprocessing for all models/analysis
The data provided is originally in a heavily-nested JSON tree-based structure. The first layer of the tree is mostly summary information about the processing (for example when the match was processed, not relevant) and match-level information (for example, total eliminations in match). Early on in the project, I realized that match-level data was too high-level and didn't capture the nuances of the game required for my research questions. I also discovered that the data had mixed game-types. Some game-types are shorter than others, resulting in bi-modal feature distributions due to the inconsistent match length. 

<img src="../imgs/pre_norm_round_length.png" alt="win ratio map" width="400"/>

To solve both these problems, I decided to "normalize" the data by round. Rather than looking at match level, I would look at the round level (each row of my data would be a round rather than a match). This also allowed the mixed-gametypes to be comparable.

<img src="../imgs/post_norm_round_length.png" alt="win ratio map" width="400"/>

### Parsing
Since the higher levels of the data-tree were essentially summaries of the deeper levels, in order to extract the information I needed I decided to start my parsing at the "source", the `'statistics'` node. I also maintained the `'user_id'` information in case it became relevant in the future. All of this preprocessing is done in `parsing/norm_round_spikesplit.py` and `preprocess/ability_remap.py`. The process is as follows:

#### Flattening
1. Dropped the all cols besides `'statistics'` and `'user_id'`
2. Flattened the stats col by 1 level
    - Removed cols from the match "perspective", such as `'totals'` and `'result'`
    - Even though `'map'` is from match perspective, I need it so I am adding it in (similar to userID)
    - Removed `'allies_onscreen'`, `'opponents_onscreen'`, and `'detections_totals'` (I think it is possible to engineer something from these columns, but this was an inconsistently collected feature so I dropped it)
3. Flattened the `'player_ids'` col by 1 level
    - Gets all the teammates and opponents, as well as maps the `'player'` to an ally (see the "Re-mapping active player" section below)
4. Flattened the `'round_info/round_totals'` cols by 1 level
    - This is where the DF expands into rounds being the rows instead of matches being the rows

#### Collecting features
The main parsing code happens during the 3rd "flattening" process (step 4 above). There are several helper functions as the data structure frequently changes. `def calculate_metrics` handles splitting the metrics based on the planting of the spike (see "Pre-Post spike plant" section below). `def calculate_longest_duration` handles calculating the longest inventory duration. `def calculate_ability_usage` handles calculating when an ability was actually used, since the data only includes if a user has an ability charge.

The rest of the code is highty specific to the feature (again due to the changing data structure). `norm_round_spikesplit.py` is extensively commented, so here I will only explain at a high-level. The goal of the parsing is to compress the data into a format that is usable for modeling. For all of the nodes, an event is marked with a timestamp when there is a change in value. For example, if you use an ability, your ability charge will go down by 1. The challenge is that not all players have the same information. Ability is restricted to the active player. However health is known across the team. Therefore, each of these data points need to be carefully handled so the resulting vector has 0s where appropriate, or does not include that feature if it is not captured. 

Since this is time-series data, I calculated averages, totals, and max change for each of the features over the period of the round. 
#### Pre-Post spike plant
As mentioned above, originally I was just taking an average, total, and max change for the entire round. I found that whether or not the spike had been planted in a round seemed to be an important feature. So, I decided to increase the "time resolution" of the data by calculating those same metrics pre-spike plant, and post-spike plant. If no spike was planted, it is all pre-spike. This required me to first parse the data to collect the spike information, then do a second pass where I assign the values accordingly.

#### Unique features
I also engineered/created some unique features. After talking with the Omnic team and doing my own research, I discovered that particular aspects"of the game could be important for my research questions, but weren’t currently being captured in the feature space. One such aspect was movement around the map. For example, some players take on roles such as "scout" where they are responsible for moving around and learning information. Other players play as "anchors", which are responsible for defending sites. I hypothized that players may have different movement levels, and I wanted to capture this at a high-level. So, I developed the `% map covered` feature which takes in the current map, available places on a that map, places the player visited, and then produces a percentage covered. I also created a feature `movement %` which provides a measure of how many regions a player is "crossing". If a player is moving back and forth between 2 areas, that would be a low value. If the player is moving across many areas, would be a high value.

#### Ability enhancement
Abilities are a huge aspect of the game. As discussed above, the raw data tracked if an ability is charged during the round (and after my pre-processing if an ability is used during the round). However, the raw data  didn't provide any information as to what that ability actually does during the game. To address this issue, I created a document `preprocess/abilities.csv` (collaboration with the Omnic team and my research) that categorizes the abilities based on their effect according to an agent's kit. 

For example, Jett is a "selfish" agent whos' abilities are fully centered around helping herself do more damage and cover more ground. Of course, she is still a great help to her team when played as an "entry fragger" (someone who runs in enemy territory and creates openings for the rest of the team). Her bailies are mapped to "damage_for_self". In contrast, a different agent called Sage is much more supportive. She is one of the few healers in the game, so her abilities are mapped to 'healing'. She can also manipulate sight lines with her smokes, so those abilities are mapped to 'crowd_control_team'. I followed this process for each agent, mapping their abilities to categories. I then used that mapping as a "look-up table" and combined it with my current dataframe to enhance the feature space (this process is done in `preprocess/ability_remap.py`) 

#### Re-mapping active player
The last major item for the data parsing is the treatment of the active player. In the raw data, the "active player" can be mapped to ally 0,1,2,3 or 4. This means that the information available for "ally1" is different between the rounds. This is bad for data consistency. If not fixed, the data would incorrectly assign 0s in places where it should really be "no information". I fixed this by re-naming the "active player" for a round to "self", and then re-mapping the remaining allies. This creates a consistent information space so each round is comparable to the other rounds, and 0s genuinely represent 0 and "no information" is not included.

#### Cleaning
The gun attributes had a miss-label, so I fixed that. And, there were a couple rows with missing information so I dropped those.
