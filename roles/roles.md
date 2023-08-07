# Roles in Valorant
The purpose of this document is to walk through the second batch of research questions related to roles in Valorant.

## Questions and additional context
- Kit = character design... an agents abilities, features, strengths, weaknesses
- In Valorant, the 19 agents are automatially organized into 4 classes: sentinels, controllers, duelists, and initiators
    - These classes have general attributes, but some are more flexible than others
        - For example, duelists are generally offensive, damage focued agents while sentinels are more supportive/defensive. Some duelist agients have a ridgid kit that can only be played offensively, while others are flexible and could be supportive despite being a "duelist".
- The Valorant community has loosely defined more specific "roles" that go beyond the pre-defined classes and resemble positions in sports
    - For example, a commonly-recommended strategy is to play as a "entry-fragger", which is a more specific role than than just "duelist".
    - More information: 
- Hypothesis: it possible to classify Valorant players into "classes" or "roles" based solely on game-play data
    - Sentinels, controllers, duelists, and initiators truly use different strategies and gameplay mechanics that can be used as features to predict the class
    - There are more specific classes, called roles, that are more infomrative to how users actually play
        - More information: 

## Is it possible to classify Valorant players into "classes" based solely on game-play data?
Why should we start with this question rather than the more interesintg role/position question? Two reasons:
1. It is an easier question with labels (supervised), that acts as a "proxy" for our harder question
2. If it is possible to classify based on gameplay, then it is more likely that our data is "good" enouguh to answer the harder question

## Data & Pre-processing
Refer to the [preprocess section](https://github.com/sophiacofone/omnic_ml/edit/main/preprocess/preprocess.md) for information on overall data preprocessing. In addition to these steps, `roles/valorant_role_data_preprocessing.py` drops some irrelevant columns (`['player','round_number']`), re-maps the true/false strings to 1 and 0, and one-hot encodes the other categorial features (`'map','self_character','ally4_character','ally1_character','ally2_character','ally3_character','opponent5_character', 'opponent6_character','opponent7_character','opponent8_character','opponent9_character','round_info_ally_side','self_longest_inv_state','self_longest_gun_primary','self_longest_gun_secondary','self_post_spike_longest_inv_state','self_pre_spike_longest_inv_state','self_post_spike_longest_gun_secondary','self_pre_spike_longest_gun_secondary','self_post_spike_longest_gun_primary','self_pre_spike_longest_gun_primary']`). I added a feature to capture round length (`['round_info_round_length']`). Finally, I encoded the valorant agents into their respective classes. `win_loss/valorant_role_data_preprocessing.py`outputs a csv per dataset, and a combined csv for all the datasets. These csvs are the input data for the modeling below. The combined dataframe has 28959 rows × 522 columns.

### Feature engineering
One of the critial "information spaces" for this analysis is abilites. In Valorant, abilities are what allow for smoke screens, increased speed, healing, infomration gathering, etc. The orignal data simply tracked if an ability was used withiout any infomration as to what that ability actually did duing the game. To fix this, I created additional features that ehnaved the ability infomration. This proved to be quite sucessful, icreasing model accuracy by 10%. Please see the [preprocess document, Ability Enhancement section](https://github.com/sophiacofone/omnic_ml/edit/main/preprocess/preprocess.md) for more inforamtion. The data presented below is using this "ehanced" feature space.

### Class Imbalance
My target was relatively balanced. Class 0 and 3 could have been higher, but it didn't seem to effect the F1 scores (all the classes were similar to predict)

<img src="../imgs/class_balance.png" alt="win ratio map" width="400"/>

## Decision Tree Model
During the first phase of the research project, I had sucess using both decision trees and logistic regression for the win/loss classification problem. Since we still care a lot about features/interpretability, I stuck with decision trees.

For more information on how deciion trees work for this data, see [win loss section](https://github.com/sophiacofone/omnic_ml/edit/main/preprocess/win_loss.md)

### Results
#### Classifying Valorant Agents: All data
I created a [results csv](https://github.com/sophiacofone/omnic_ml/edit/main/preprocess/win_loss.md) of the features and their importances. The magnitude indicates the "importance" of the feature.

##### Metrics: Tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 100%      |
| Test Accuracy  | 96%      |
| Train F1       | 100%      |
| Test F1        | 96%      |
##### Metrics: After pruning (max 20 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 96%      |
| Test Accuracy  | 93%      |
| Train F1       | 93%      |
| Test F1        | 93%      |
##### Top 20 important features combined
| Feature                                  | Importance              |
|------------------------------------------|-------------------------|
|   user_id                                |   0.10913292461324700   |
|   self_pre_spike_total_ability_usage_3   |   0.08143161322120570   |
|   pre_spike_information_used             |   0.06695606672304130   |
|   pre_spike_damage_for_self_used         |   0.060533953589459200  |
|   pre_spike_crowd_control_vision_used    |   0.04961458316276060   |
|   pre_spike_damage_for_team_used         |   0.03172291146035310   |
|   pre_spike_shield_used                  |   0.028227662504068500  |
|   self_pre_spike_total_ability_usage_1   |   0.023795308739751200  |
|   ally2_character_killjoy                |   0.022779482397761600  |
|   pre_spike_crowd_control_mobility_used  |   0.02207363059751190   |
|   pre_spike_crowd_control_general_used   |   0.020619009885599700  |
|   opponent6_character_jett               |   0.01862002612811630   |
|   opponent8_character_raze               |   0.017376152521623900  |
|   self_pre_spike_total_ability_usage_2   |   0.01732240326187030   |
|   self_pre_spike_map_covered             |   0.01645849301757500   |
|   pre_spike_healing_used                 |   0.014476640390032500  |
|   ally3_character_skye                   |   0.012524630737437000  |
|   opponent8_character_cypher             |   0.01058859294778480   |
|   ally3_character_killjoy                |   0.010326296666824100  |
|   ally3_pre_spike_avg_health             |   0.010163645082855500  |

#### Classifying Valorant Agents: No user ID
##### Metrics: Tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 100%      |
| Test Accuracy  | 96%      |
| Train F1       | 100%      |
| Test F1        | 96%      |
##### Metrics: After pruning (max 18 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 93%      |
| Test Accuracy  | 91%      |
| Train F1       | 93%      |
| Test F1        | 91%      |
##### Top 20 important features combined
| Feature                                  | Importance              |
|------------------------------------------|-------------------------|
|   self_pre_spike_total_ability_usage_3   |   0.08567188580590630   |
|   pre_spike_information_used             |   0.07186504042188400   |
|   pre_spike_damage_for_self_used         |   0.0627606069213998    |
|   pre_spike_crowd_control_vision_used    |   0.052953592286810600  |
|   self_pre_spike_total_ability_usage_1   |   0.04155741290154830   |
|   pre_spike_shield_used                  |   0.04054615642613230   |
|   pre_spike_damage_for_team_used         |   0.03524908383761920   |
|   pre_spike_crowd_control_general_used   |   0.023149207137497800  |
|   pre_spike_crowd_control_mobility_used  |   0.02309451725061980   |
|   ally2_character_killjoy                |   0.021884859563280800  |
|   self_pre_spike_total_ability_usage_2   |   0.01793597110499090   |
|   self_pre_spike_map_covered             |   0.017656919047466100  |
|   ally1_pre_spike_total_health_loss      |   0.0163318391073393    |
|   pre_spike_healing_used                 |   0.01483720263818700   |
|   map_Pearl                              |   0.014450950938648300  |
|   opponent7_character_killjoy            |   0.014417562505240200  |
|   map_Bind                               |   0.014362167789729800  |
|   ally4_character_reyna                  |   0.01399350825386680   |
|   ally3_character_skye                   |   0.012736180379580200  |
|   ally4_pre_spike_avg_health             |   0.011279023511655200  |

#### Classifying Valorant Agents: No user ID, opponent characters
##### Metrics: Tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 100%      |
| Test Accuracy  | 85%      |
| Train F1       | 100%      |
| Test F1        | 85%      |
##### Metrics: After pruning (max 15 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 88%      |
| Test Accuracy  | 82%      |
| Train F1       | 82%      |
| Test F1        | 82%      |
##### Top 20 important features combined
| Feature                                  | Importance              |
|------------------------------------------|-------------------------|
|   self_pre_spike_total_ability_usage_3   |   0.10503614895710700   |
|   pre_spike_information_used             |   0.10042461148492200   |
|   pre_spike_damage_for_self_used         |   0.07241439102127120   |
|   self_pre_spike_total_ability_usage_1   |   0.06621886948176720   |
|   pre_spike_crowd_control_vision_used    |   0.06124320682331280   |
|   pre_spike_shield_used                  |   0.04735179217110950   |
|   pre_spike_damage_for_team_used         |   0.045136745364399700  |
|   pre_spike_crowd_control_mobility_used  |   0.03409119894096020   |
|   ally1_pre_spike_avg_health             |   0.033603899351380600  |
|   pre_spike_crowd_control_general_used   |   0.02745621174533880   |
|   self_pre_spike_total_ability_usage_2   |   0.0262312664059472    |
|   self_pre_spike_map_covered             |   0.02389487511043900   |
|   pre_spike_healing_used                 |   0.023156750118795100  |
|   map_Pearl                              |   0.02181648212301020   |
|   ally3_pre_spike_avg_health             |   0.017360431756040300  |
|   self_pre_spike_avg_ammo_reserve        |   0.016608830058298600  |
|   map_Bind                               |   0.015792331734720600  |
|   map_Fracture                           |   0.013073441633814500  |
|   map_Lotus                              |   0.012167165309781200  |
|   self_pre_spike_max_loadout_value_loss  |   0.01164289305912030   |
|   self_pre_spike_max_loadout_value_loss  |   0.01164289305912030   |

#### Classifying Valorant Agents: No user ID, opponent characters, or map
##### Metrics: Tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 98%      |
| Test Accuracy  | 84%      |
| Train F1       | 98%      |
| Test F1        | 84%      |
##### Metrics: After pruning (max 20 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 84%      |
| Test Accuracy  | 79%      |
| Train F1       | 79%      |
| Test F1        | 79%      |
##### Top 20 important features combined
| Feature                                    | Importance              |
|--------------------------------------------|-------------------------|
|   self_pre_spike_total_ability_usage_3     |   0.11793070675455400   |
|   pre_spike_information_used               |   0.10882476819712400   |
|   pre_spike_damage_for_self_used           |   0.07879679348356660   |
|   self_pre_spike_total_ability_usage_1     |   0.0733969845819977    |
|   pre_spike_crowd_control_vision_used      |   0.06811062161769070   |
|   pre_spike_shield_used                    |   0.05565653121651830   |
|   pre_spike_damage_for_team_used           |   0.04328348937400650   |
|   pre_spike_crowd_control_mobility_used    |   0.03978952113607500   |
|   self_pre_spike_total_ability_usage_2     |   0.03617835193917490   |
|   pre_spike_healing_used                   |   0.030047390973284700  |
|   self_pre_spike_map_covered               |   0.029461361280699100  |
|   pre_spike_crowd_control_general_used     |   0.028862215988680100  |
|   ally1_pre_spike_total_health_loss        |   0.021342729444774900  |
|   ally3_pre_spike_avg_health               |   0.020863498232882200  |
|   self_pre_spike_avg_ammo_reserve          |   0.019674341205390200  |
|   self_pre_spike_movement_metric           |   0.013212286257612300  |
|   self_pre_spike_total_loadout_value_loss  |   0.01189121807315640   |
|   self_pre_spike_total_ability_usage_4     |   0.010827979729116900  |
|   ally1_pre_spike_avg_health               |   0.009926334094066230  |
|   ally4_pre_spike_avg_health               |   0.009643626625648180  |


## Are there other classes/roles that are more informative of how users actually play?
After hypothesising what these roles might be and doing strategic feature engineering to try and provide as much of that information to my models as I could, i decided to attempt clustering the data. My reasoning was, if I can form clusters with the data, we can see if those clusters align with some of the roles we might expect.

This ended up being challenging. Despte having classification sucess eariler in the project, the data was not forming good clusters with the methods I tried. I tried using PCA, Kernel PCA, and T-SNE. None of these methods created easily defineable clusters. My reasoning behind why this might be happening is the nature of the data. The data is sparse, with many features being corelated and categorical, whcih could be challening for distance-based clistering methods.

I tried significantly reducing the feature space (using features deemed important from the previous work done with the decision trees) to see if that helped. 