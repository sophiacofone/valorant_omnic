# Winning Rounds in Valorant
The purpose of this document is to walk through the first batch of research questions related to winning rounds in Valorant.

## Questions and additional context
- To win a match of Valorant, you need to win 13 rounds (5 rounds if the game-type is swiftplay).
- There are several ways to win a round. The options are dependent on if you are attacking or defending.
    - If you are on the attacking side, you win by either planting the spike on the opposing side and it detonates, or you eliminate all opponents.
    - If you are on the defending side, you win by either deactivating a planted spike, eliminating all opponents, or just surviving without the spike being planted.
- Hypothesis: depending on if you are attacker or defender, and if the spike has been planted or not, your strategy will change
    - If you are an attacker and the spike is not planted, you should focus on planting the spike.
    - If you are an attacker and the spike is planted, all teammates should focus on defending the spike and killing the other team.
    - If you are a defender and the spike is not planted, the team should focus on kills and defending the site.
     If you are a defender and the spike is planted, the team should focus on kills/deactivating the spike.

## Are there certain characters, maps that are more likely to lead to a win?
A well-designed game would not have one map that was significantly easier to win on, or one character that was dominant over the rest. To assess this, I grouped the data by feature and calculated the win-ratio.

<img src="../imgs/wr_map.png" alt="win ratio map" width="400"/> <img src="../imgs/wr_agent.png" alt="win ratio agent" width="400"/>
<img src="../imgs/wr_ad.png" alt="win ratio attack defend" width="400"/> <img src="../imgs/wr_spike.png" alt="win ratio spike" width="400"/>

I found the map types and agents to be balanced. According to this data, there does not seem to be any one map or agent that was "easier" to win with. I also checked the win/loss ratio for attackers vs defenders, as well as spike planted vs not. Like map and agent, these visuals indicate that it is not "easier" to be an attacker or defender, or that it's always "better" to plant the spike rather than just eliminating the whole opposing team.

This is not a surprising result. In fact, this indicates that what it takes to win a round of Valorant is more strategic and subtle than simply picking the best agent (for example). Individual players may still have maps or agents that consistently give them better outcomes. However, this is likely due to preference/practice rather than the agent/map itself being inherently better.

Refer to the [EDA code](https://github.com/sophiacofone/omnic_ml/blob/main/EDA/eda.py) for more information on the EDA process and how these visuals were generated.

## What should players focus on to win a round of Valorant?
To answer this question, one strategy is to build a classifier that can accuratley predict wins and losses. Then, we can examine the feature importances/coefficents and determine which factors are most influential in determining match outcomes. Since we care less about the actual prediction and more about the features, a good choice would be to work with with models like decision trees and logistic regression since these models can be easily interpreted.

### Data & Preprocessing
Refer to the [preprocess section](https://github.com/sophiacofone/omnic_ml/edit/main/preprocess/preprocess.md) for information on overall data preprocessing. In addition to these steps, `win_loss/win_loss_data_preprocessing.py` drops some irrelevant columns (`['player','round_number']`), re-maps the true/false strings to 1 and 0, and one-hot encodes the other categorial features (`'map','self_character','ally4_character','ally1_character','ally2_character','ally3_character','opponent5_character', 'opponent6_character','opponent7_character','opponent8_character','opponent9_character','round_info_ally_side','self_longest_inv_state','self_longest_gun_primary','self_longest_gun_secondary','self_post_spike_longest_inv_state','self_pre_spike_longest_inv_state','self_post_spike_longest_gun_secondary','self_pre_spike_longest_gun_secondary','self_post_spike_longest_gun_primary','self_pre_spike_longest_gun_primary']`). Finally, I added a feature to capture round length (`['round_info_round_length']`). `win_loss/win_loss_data_preprocessing.py`outputs a csv per dataset, and a combined csv for all the datasets. These csvs are the input data for the modeling below. The combined dataframe has 28959 rows × 542 columns.

Please see [win_loss_data_preprocessing](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/win_loss_data_preprocessing.py) for the w/l-specific preprocessing code.

## Logistic Regression Model
Logistic regression is usually a good starting point for binary classification problems. As mentioned above, its primary "pro" is that it is simple and easy to interpret. It is also fast, easy to implement, and isn't really prone to overfitting. However, its main downside is that it can be **too** simple. It assumes the data has a linear relationship with the outcome. Therefore, it works best when the data is linearly separable. Logistic regression struggles to capture complex relationships due to this linearity assumption.

The "goal" of logistic regression is to predict one of two outcomes based on the input features. The logistic function (sigmoid function) maps any number to a value between 0 and 1 (good for probabilities).

Logistic function: -σ(z) = 1 / (1 + e^(-z))
- σ(z) is the output (probability) between 0 and 1.
- e is the base of the natural logarithm, approximately equal to 2.71828.
- z is the linear combination of the input features and their associated weights.
    - z = w0 * x0 + w1 * x1 + w2 * x2 + ... + wn * xn
    - w0, w1, w2, ..., wn are the coefficients (weights) associated with each feature.
    - x0, x1, x2, ..., xn are the corresponding feature values.

During training, the model finds the best values for the coefficients (weights) that minimize the error between the predicted probabilities and the actual class labels in the training data.

Please see [win_loss_model](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/win_loss_model.py) for the modeling code.

### Class Imbalance
Log reg is sensitive to class imbalance. My first step was to ensure my target was balanced.

<img src="../imgs/g_won.png" alt="win ratio map" width="400"/>

### Feature Selection
Since we are ultimately interested in the **features** of this model, I decided to incorporate feature selection into this process. I first trained a logistic regression model using L1 regularization. L1 regularization has the benefit of driving some feature weights to zero, effectively excluding them from the model. I decided to further eliminate features by only including features with coefs that were above the median value. Then, I used those features to create a new model using L2 regularization. L2 regularization prevents overfitting and is less sensitive to outliers than L1. In both cases, I used cross validation to tune the "C" hyper parameter (controls the strength of the regularization).

### Results
| Data Sub-Set | Main Selected Features                                            |
|--------------|--------------------------------------------------------------|
| All          | Deaths, Health                                               |
| No Deaths    | Eliminations                                                 |
| Attack       | Deaths, Health, Ability use                                  |
| Defend       | Deaths, Health, Ammo use          |
| Pre-Spike    | Deaths, Health, Gun use                                      |
| Post-Spike   | Deaths, Health, Credits           |
####  What should player's focus on to win a round of Valorant?: All data
Refer to [results section](https://github.com/sophiacofone/omnic_ml/tree/main/win_loss/logreg_csv_feature_results) for all of the generated feature coefs. The magnitude indicates the "importance" of the feature.

Please see [win_loss_applied](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/win_loss_applied.py) for the modeling inference.

##### Metrics
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 95%      |
| Test Accuracy  | 95%      |
| Train F1       | 95%      |
| Test F1        | 95%      |
<img src="logreg_csv_feature_results/confusion_mat_all.png" alt="" width="400"/>

##### Top 20 important features combined
|   Feature                                 |   coef                 |
|-------------------------------------------|------------------------|
|   all_opponent_dead                       |    0.9882489094383900  |
|   ally1_post_spike_deaths                 |   -0.9385987971353920  |
|   ally1_pre_spike_deaths                  |   -0.898857198849475   |
|   self_pre_spike_avg_health               |   0.8439591098285130   |
|   ally2_pre_spike_deaths                  |   -0.8429755584496300  |
|   ally3_pre_spike_deaths                  |   -0.7493854195499950  |
|   opponent3_pre_spike_deaths              |   0.7336340167650350   |
|   ally4_post_spike_deaths                 |   -0.7312162146760140  |
|   opponent1_pre_spike_deaths              |   0.7242236307036010   |
|   ally4_pre_spike_deaths                  |   -0.7183141248410150  |
|   opponent4_pre_spike_deaths              |   0.712433619067903    |
|   opponent2_pre_spike_deaths              |   0.6835489056728720   |
|   self_longest_inv_state_melee            |   -0.6757866540937680  |
|   opponent0_pre_spike_deaths              |   0.6723183751844650   |
|   all_ally_dead                           |   -0.6680610650312690  |
|   self_pre_spike_longest_inv_state_melee  |   0.6370944054678590   |
|   self_pre_spike_deaths                   |   -0.6368605941682240  |
|   ally3_post_spike_deaths                 |   -0.6307989473994810  |
|   ally3_post_spike_total_health_loss      |   0.6061965632616760   |
|   self_post_spike_total_health_loss       |   -0.5753664427197250  |

These features indicate that from an overall perspective, your team not dying, your opponents dying, and health are the most important predictor for winning rounds of Valorant. This may seem obvious, but in Valorant there are multiple ways to win with elims/deaths being only one of them. These results could justify playing more "defensively", i.e. not dying versus trying to get lots of elims by taking risky moves.

#### No deaths?
I wanted to also see what without using deaths as a feature, could the model still predict well and what features it would pick.

##### Metrics
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 95%      |
| Test Accuracy  | 94%      |
| Train F1       | 95%      |
| Test F1        | 94%      |
<img src="logreg_csv_feature_results/confusion_mat_nod.png" alt="" width="400"/>

##### Top 20 important features combined - no deaths
|   Feature                                 |   Coef                 |
|-------------------------------------------|------------------------|
|   ally1_post_spike_deaths                 |   -1.6703996301272200  |
|   opponent1_pre_spike_elims               |   -1.2798674405305800  |
|   ally3_pre_spike_elims                   |   1.2300565204608700   |
|   ally2_pre_spike_elims                   |   1.18446367741373     |
|   ally1_pre_spike_elims                   |   1.1376096807716800   |
|   ally4_pre_spike_elims                   |   1.110000294505390    |
|   opponent3_pre_spike_elims               |   -1.0996632153713400  |
|   opponent2_pre_spike_elims               |   -1.0983541264935200  |
|   opponent4_pre_spike_elims               |   -1.0707871269108400  |
|   self_pre_spike_elims                    |   1.0310610615631100   |
|   all_opponent_dead                       |   0.9966485067026910   |
|   opponent0_pre_spike_elims               |   -0.9925280504564480  |
|   opponent2_post_spike_elims              |   -0.9839914759272690  |
|   opponent3_post_spike_elims              |   -0.9092449766110730  |
|   self_pre_spike_avg_health               |   0.8348623251219470   |
|   opponent1_post_spike_elims              |   -0.8298465815564710  |
|   ally3_post_spike_elims                  |   0.806710343293057    |
|   self_longest_inv_state_melee            |   -0.7441413663323180  |
|   self_pre_spike_longest_inv_state_melee  |   0.714229161153148    |
|   all_ally_dead                           |   -0.6919451999149420  |
|   opponent4_post_spike_elims              |   -0.6881325572874230  |

The model does still predict well, but this time it appears to be focusing on elims rather than deaths. 

####  What should player's focus on to win a round of Valorant?: Attacking vs Defending
Now that we have an idea of what it takes to win a round of valorant from an overall level, I thought it would be interesting to explore if these features change depending on some additional criteria. I decided to use a "stratified analysis" approach, where I divide my data into groups and separately perform my analysis and investigate the differences (if there are any).

First, I divided my data into "attacker" and "defender" (I removed the unknown rows).

<img src="../imgs/g_ally.png" alt="win ratio map" width="400"/>

Then, I re-ran my process using `win_loss/csv/'wl_alldf_prepro_data_attack.csv` and `win_loss/csv/'wl_alldf_prepro_data_defend.csv` producing `win_loss/df_coefs_logreg_attack.csv` and `win_loss/df_coefs_logreg_defenc.csv`.

Please see [stratified_df](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/stratified_df.py) for the devided dataframde code.

##### Metrics
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 97%      |
| Test Accuracy  | 96%      |
| Train F1       | 97%      |
| Test F1        | 96%      |
<img src="logreg_csv_feature_results/confusion_mat_attack.png" alt="" width="400"/>

##### Top 20 important features combined: Attack
| Feature                                 | Coef                   |
| --------------------------------------- | ---------------------- |
|   ally1_post_spike_deaths               |   -1.6703996301272200  |
|   ally4_post_spike_deaths               |   -1.3721565846202400  |
|   ally3_post_spike_deaths               |   -1.2383865192985900  |
|   ally2_post_spike_deaths               |   -1.207032117040700   |
|   round_info_round_length               |   -1.1651121582417600  |
|   opponent3_post_spike_deaths           |   1.1507132924721000   |
|   all_opponent_dead                     |   1.1307786860961700   |
|   self_post_spike_deaths                |   -1.0733820756241300  |
|   post_spike_crowd_control_vision_used  |   1.0404368913979500   |
|   ally3_post_spike_total_health_loss    |   0.9989532836984400   |
|   opponent1_pre_spike_deaths            |   0.9508902263537160   |
|   ally1_pre_spike_deaths                |   -0.9486844033856280  |
|   opponent2_pre_spike_deaths            |   0.9350685482106680   |
|   opponent0_pre_spike_deaths            |   0.92489327480526     |
|   opponent3_pre_spike_deaths            |   0.9223488365584660   |
|   self_post_spike_max_health_loss       |   -0.8862837203866290  |
|   opponent4_post_spike_deaths           |   0.8185005839381280   |
|   opponent4_pre_spike_deaths            |   0.7785656109722340   |
|   all_ally_dead                         |   -0.7726353504666180  |
|   ally4_pre_spike_deaths                |   -0.7697123673928790  |

Similar to the analysis above, we see deaths/not dying as the best thing to focus on. However, we do see some ability use here.

##### Metrics
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 96%      |
| Test Accuracy  | 95%      |
| Train F1       | 96%      |
| Test F1        | 96%      |
<img src="logreg_csv_feature_results/confusion_mat_defend.png" alt="" width="400"/>

##### Top 20 important features combined: Defend
| Feature                                 | Coef                   |
| --------------------------------------- | ---------------------- |
|   ally1_post_spike_deaths                     |   -1.6703996301272200  |
|   self_post_spike_total_health_loss           |   -1.4048513997774400  |
|   all_opponent_dead                           |   1.2895060099724600   |
|   self_longest_gun_primary_phantom            |   -1.178260381814890   |
|   self_longest_inv_state_melee                |   -1.1680928617356300  |
|   ally3_post_spike_total_health_loss          |   1.143035643510180    |
|   self_post_spike_total_ammo_reserve_loss     |   1.0866869332501200   |
|   self_pre_spike_longest_gun_primary_phantom  |   1.0694273048158000   |
|   self_pre_spike_longest_inv_state_melee      |   1.0603282172138800   |
|   post_spike_crowd_control_vision_used        |   1.0171755104521000   |
|   post_spike_damage_for_team_used             |   0.870745813765667    |
|   ally4_post_spike_elims                      |   0.838567585409549    |
|   ally4_post_spike_deaths                     |   -0.8037460538647530  |
|   self_pre_spike_avg_health                   |   0.8026429247463740   |
|   self_pre_spike_longest_gun_primary_odin     |   0.7753015792331070   |
|   opponent3_pre_spike_elims                   |   -0.7384452951607550  |
|   self_post_spike_max_ammo_reserve_loss       |   -0.7345382915075390  |
|   self_pre_spike_deaths                       |   -0.7220809635938230  |
|   ally2_pre_spike_deaths                      |   -0.7113390993033230  |
|   ally1_post_spike_deaths                     |   -0.7102710207246480  |
|   ally1_post_spike_total_health_loss          |   0.7024308528414130   |
|   opponent1_pre_spike_deaths                  |   0.6753087258632680   |

Here, we see some differences. Deaths/health is still the highest, but we start to see ammo, gun usage, ability usage, and elims.

####  What should player's focus on to win a round of Valorant?: Pre vs Post Spike plant

##### Metrics
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 92%      |
| Test Accuracy  | 91%      |
| Train F1       | 92%      |
| Test F1        | 91%      |
<img src="logreg_csv_feature_results/confusion_mat_pres.png" alt="" width="400"/>

##### Top 20 important features combined: Pre-Spike
| Feature                                        | Coef                    |
| ---------------------------------------------- | ----------------------- |
|   ally1_post_spike_deaths                      |   -1.6703996301272200   |
|   all_opponent_dead                            |   1.6855836963388600    |
|   all_ally_dead                                |   -1.6139346397139800   |
|   self_longest_gun_secondary_ghost             |   -0.9951241240938690   |
|   self_pre_spike_longest_gun_secondary_ghost   |   0.9783090842880590    |
|   opponent1_pre_spike_deaths                   |   0.5863409737695190    |
|   opponent3_pre_spike_deaths                   |   0.5721287494268150    |
|   opponent4_pre_spike_deaths                   |   0.5464931983882780    |
|   ally1_pre_spike_deaths                       |   -0.5261706588499960   |
|   ally2_pre_spike_deaths                       |   -0.4915017959346170   |
|   opponent2_pre_spike_deaths                   |   0.4696511872098220    |
|   opponent0_pre_spike_deaths                   |   0.46936088048022200   |
|   map_unknown                                  |   -0.4588306637063800   |
|   self_pre_spike_longest_gun_primary_judge     |   0.44560968659553200   |
|   ally4_pre_spike_deaths                       |   -0.44503439143329000  |
|   ally3_pre_spike_deaths                       |   -0.4195541890113650   |
|   self_longest_inv_state_primary               |   0.41246459682815700   |
|   self_pre_spike_avg_health                    |   0.4050623954135690    |
|   self_longest_gun_primary_operator            |   -0.3964144448018260   |
|   self_longest_gun_primary_judge               |   -0.39233111767187600  |
|   self_pre_spike_longest_gun_primary_operator  |   0.3753082448642650    |
|   opponent1_pre_spike_deaths                   |   0.6753087258632680    |

We still see deaths/health as big predictors, but with a lot more gun-related features.

##### Metrics
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 89%      |
| Test Accuracy  | 88%      |
| Train F1       | 90%      |
| Test F1        | 89%      |
<img src="logreg_csv_feature_results/confusion_mat_posts.png" alt="" width="400"/>

##### Top 20 important features combined: Post-Spike
| Feature                                    | Coef                    |
| ------------------------------------------ | ----------------------- |
|   ally1_post_spike_deaths                  |   -1.6703996301272200   |
|   all_opponent_dead                        |   1.988560477374960     |
|   all_ally_dead                            |   -1.8306075999282400   |
|   ally3_post_spike_total_health_loss       |   0.5748347432541790    |
|   ally1_post_spike_deaths                  |   -0.5371718797330310   |
|   self_post_spike_total_health_loss        |   -0.5158301434749740   |
|   self_post_spike_avg_credits              |   0.5001711474262690    |
|   ally4_post_spike_deaths                  |   -0.4193000369542100   |
|   self_post_spike_total_ammo_reserve_loss  |   0.40222373910636200   |
|   opponent1_post_spike_elims               |   -0.39497056505775300  |
|   spike_planted                            |   -0.3861514073713010   |
|   ally1_post_spike_total_health_loss       |   0.34890964274876600   |
|   self_post_spike_longest_inv_state_none   |   -0.3407263419872230   |
|   ally4_post_spike_avg_health              |   0.31998347356295800   |
|   self_post_spike_total_ability_usage_4    |   0.305710216622554     |
|   ally3_post_spike_deaths                  |   -0.2944827300192010   |
|   ally3_post_spike_elims                   |   0.29382870545559900   |
|   post_spike_damage_for_team_used          |   0.2933489639029770    |
|   opponent4_post_spike_headshots           |   -0.29046241197795200  |
|   ally1_post_spike_avg_health              |   0.27804356679877700   |
|   self_post_spike_max_ammo_reserve_loss    |   -0.27191360986471900  |
|   opponent1_pre_spike_deaths               |   0.6753087258632680    |

We still see deaths/health as big predictors, but also credits (for the first time), ability usage, headshots.

## Decision Tree Model
Even though I achieved good results with the simple logistic regression models, I wanted to also explore a non-linear classifier like decision trees. Decision trees work by recursively partitioning the input space into smaller regions, and making predictions based on the majority class or average value of the target variable in that region. The decision tree starts by selecting the best feature to split the data. The "best" is defined by the feature that maximizes the separation of the classes. After the data is split, each subset becomes a new "node" in the tree. This process continues until a stopping criteria is met. Once the tree is finished building, we can then assign labels to the terminal nodes.

Again, since we are most interested in the features used for the classification, this model is a good choice. Decision trees are prone to overfitting, but we can overcome that issue via pruning. Decision trees also handle mixed variable types (categorical and continuous) well, which is ideal for our problem space.

Decision trees commonly use the Gini impurity to "split the data". Gini impurity is a measure of the degree of impurity or disorder in a set of data.  

Gini for a node
- For a given node in the decision tree that contains 'N' data points belonging to 'K' different classes Gini(node) = 1 - Σ (p_i)^2
    - 'p_i': the proportion of data points belonging to class 'i' in the node.

Gini for a split
- Weighted average of the Gini impurities of the child nodes created by the split
- Gini(split) = (N_left / N_total) * Gini(left) + (N_right / N_total) * Gini(right)
    - 'N_left' is the number of data points in the left child node after the split
    - 'N_right' is the number of data points in the right child node after the split.
    - 'N_total' is the total number of data points in the current node.
    - 'Gini(left)' is the Gini impurity of the left child node.
    - 'Gini(right)' is the Gini impurity of the right child node.

I chose to use GridSearchCV to tune the min_samples_leaf and min_samples_split parameters. min_samples_leaf specifies the minimum number of samples required to be at a leaf node (the terminal nodes of the tree). This helps prevent overfitting by creating a simpler tree with less splits (for large values of min_samples_leaf). 
min_samples_split specifies the minimum number of samples required to split an internal node during the tree-building process. Larger values of min_samples_split can help prevent overfitting by making it harder for the tree to create splits with small subsets of data.

Lastly, (as previously mentioned) I experimented with pruning (setting a maximum depth of the tree) to avoid overfitting.

Please see [win_loss_model](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/win_loss_model.py) for the modeling code.

### Results
| Data Sub-Set | Main Selected Features                                            |
|--------------|--------------------------------------------------------------|
| All          | Deaths, Health, Credits, Map % covered                                         |
| No Deaths    | Health, Eliminations                                                 |
| Attack       | Deaths, Health, Credits, Eliminations
| Defend       | Deaths, Health, Spike_time, Credits |
| Pre-Spike    | Deaths, Health, Spike_time, Movement %                                      |
| Post-Spike   | Deaths, Health, Credits, Assists            |

####  What should player's focus on to win a round of Valorant?: All data
As before, please refer to [results section](https://github.com/sophiacofone/omnic_ml/tree/main/win_loss/dtree_csv_feature_results) for all of the generated feature importances.

Please see [win_loss_applied](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/win_loss_applied.py) for the modeling inference.

##### Metrics: Tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 100%      |
| Test Accuracy  | 94%      |
| Train F1       | 100%      |
| Test F1        | 94%      |
##### Metrics: More pruning (max 5 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 90%      |
| Test Accuracy  | 90%      |
| Train F1       | 90%      |
| Test F1        | 90%      |
<img src="dtree_csv_feature_results/confusion_mat_all.png" alt="" width="400"/>

##### Top 20 important features combined
| Feature                              | Importance               |
|--------------------------------------|--------------------------|
|   all_opponent_dead                  |   0.5622840678490050     |
|   all_ally_dead                      |   0.2273113477972850     |
|   self_post_spike_total_health_loss  |   0.07888739556467370    |
|   self_pre_spike_total_health_loss   |   0.05253737646689780    |
|   ally2_pre_spike_deaths             |   0.02579466734878280    |
|   self_post_spike_avg_credits        |   0.02201917412002120    |
|   ally1_pre_spike_deaths             |   0.011124641299069800   |
|   ally4_post_spike_max_health_loss   |   0.0037713620579577200  |
|   self_post_spike_map_covered        |   0.0026900558208574400  |
|   ally2_post_spike_assists           |   0.002209893101311710   |
|   self_pre_spike_headshots           |   0.0017404518961157800  |
|   ally2_post_spike_elims             |   0.0016728803101759800  |
|   ally4_post_spike_avg_health        |   0.0014053368746943300  |
|   ally3_post_spike_avg_health        |   0.0011603012640771900  |
|   opponent0_post_spike_elims         |   0.0010885228448278200  |
|   ally4_pre_spike_total_health_loss  |   0.0010772724598542700  |
|   ally3_pre_spike_avg_health         |   0.0007096746058738760  |
|   self_pre_spike_total_shield_loss   |   0.0006163000584282590  |
|   ally4_pre_spike_max_health_loss    |   0.0006153740632695090  |
|   self_pre_spike_assists             |   0.0005510815716306480  |

Similar to the logistic regression model, these features indicate that from an overall perspective your team not dying, your opponents dying, and health are the most important predictors for wining rounds of Valorant. After those features, the model starts using credits, % map covered, assists, and elims. 

#### No deaths?
I wanted to also see what without using deaths as a feature, could the model still predict well and what features it would pick.

##### Metrics: Tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 97%      |
| Test Accuracy  | 90%      |
| Train F1       | 97%      |
| Test F1        | 90%      |
##### Metrics: More pruning (max 5 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 81%      |
| Test Accuracy  | 75%      |
| Train F1       | 81%      |
| Test F1        | 74%      |
<img src="dtree_csv_feature_results/confusion_mat_nodeaths.png" alt="" width="400"/>

##### Top 20 important features combined - no deaths
| Feature                               | Importance               |
|---------------------------------------|--------------------------|
|   self_post_spike_total_health_loss   |   0.31582639606061500    |
|   self_pre_spike_total_health_loss    |   0.28875960521342700    |
|   self_post_spike_avg_credits         |   0.08891139935371020    |
|   ally2_pre_spike_elims               |   0.0477550661912334     |
|   ally3_pre_spike_elims               |   0.04187291851682670    |
|   ally4_pre_spike_elims               |   0.028845329618753100   |
|   self_post_spike_avg_health          |   0.02492653404779900    |
|   ally1_pre_spike_elims               |   0.021933599318584600   |
|   round_info_ally_side_attacker       |   0.01761772679865010    |
|   self_post_spike_max_health_loss     |   0.014430371014569700   |
|   ally1_post_spike_total_health_loss  |   0.011119934315616500   |
|   self_post_spike_assists             |   0.00994232445960613    |
|   opponent3_post_spike_headshots      |   0.009883189105565110   |
|   opponent1_post_spike_elims          |   0.00881370479226952    |
|   opponent2_pre_spike_elims           |   0.008662255585700720   |
|   opponent4_post_spike_elims          |   0.008243363174679310   |
|   ally4_post_spike_max_health_loss    |   0.0065081302955400800  |
|   ally2_post_spike_elims              |   0.0062549089924971000  |
|   ally4_post_spike_elims              |   0.005901501801658560   |
|   opponent2_post_spike_assists        |   0.00549376100049144    |

This model does not predict as well after pruning. After Health loss, this model also focuses on elims.

####  What should player's focus on to win a round of Valorant?: Attacking vs Defending
Here, I stick with the same "stratified analysis" approach.

Please see [stratified_df](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/stratified_df.py) for the devided dataframde code.

##### Metrics: Attack, tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 99%      |
| Test Accuracy  | 94%      |
| Train F1       | 99%      |
| Test F1        | 94%      |
##### Metrics: Attack, pruning (max 5 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 92%      |
| Test Accuracy  | 92%      |
| Train F1       | 92%      |
| Test F1        | 92%      |
<img src="dtree_csv_feature_results/confusion_mat_attack.png" alt="" width="400"/>

##### Top 20 important features combined: Attack
| Feature                                  | Importance                |
|------------------------------------------|---------------------------|
|   all_ally_dead                          |   0.49676728007159500     |
|   all_opponent_dead                      |   0.22231105975151200     |
|   self_pre_spike_total_health_loss       |   0.11095798321942200     |
|   self_post_spike_avg_credits            |   0.06074763365712760     |
|   self_post_spike_deaths                 |   0.046578269731772500    |
|   ally4_pre_spike_deaths                 |   0.009963608972048250    |
|   ally1_post_spike_total_health_loss     |   0.009660135432341860    |
|   ally4_post_spike_total_health_loss     |   0.007844282031773470    |
|   opponent1_post_spike_first_bloods      |   0.007705316271720350    |
|   ally1_pre_spike_elims                  |   0.00674041611910913     |
|   ally3_post_spike_deaths                |   0.006174695106967520    |
|   self_pre_spike_headshots               |   0.0031599007867357000   |
|   user_id                                |   0.002311614953270460    |
|   ally1_pre_spike_total_health_loss      |   0.002068224104462050    |
|   opponent1_ultimate_usage               |   0.0018453999154695400   |
|   ally3_pre_spike_avg_health             |   0.0009588967303739750   |
|   ally1_post_spike_elims                 |   0.0009396665375589360   |
|   self_pre_spike_movement_metric         |   0.0008406030088691890   |
|   self_post_spike_max_ammo_reserve_loss  |   0.0008040981522628090   |

Similar to the analysis above, we see deaths/not dying/health as the best thing to focus on. However, we do see credits, elims, ability use, movement, and ammo.

##### Metrics: Defend, tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 99%      |
| Test Accuracy  | 93%      |
| Train F1       | 99%      |
| Test F1        | 93%      |
##### Metrics: Defend, pruning (max 5 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 87%      |
| Test Accuracy  | 87%      |
| Train F1       | 88%      |
| Test F1        | 88%      |
<img src="dtree_csv_feature_results/confusion_mat_defend.png" alt="" width="400"/>

##### Top 20 important features combined: Defend
| Feature                                    | Importance               |
|--------------------------------------------|--------------------------|
|   all_ally_dead                            |   0.5424268289601420     |
|   all_opponent_dead                        |   0.22276433599465900    |
|   self_pre_spike_total_health_loss         |   0.0935347860128104     |
|   self_post_spike_total_health_loss        |   0.04721171715932880    |
|   spike_time                               |   0.03724128313832150    |
|   opponent1_pre_spike_deaths               |   0.012197742290977600   |
|   ally4_post_spike_max_health_loss         |   0.010542252455186100   |
|   ally1_post_spike_total_health_loss       |   0.007398771093428020   |
|   ally2_post_spike_deaths                  |   0.006326503928391290   |
|   self_post_spike_avg_credits              |   0.004334172365594340   |
|   self_post_spike_total_shield_loss        |   0.002563361824531640   |
|   ally4_pre_spike_avg_health               |   0.00218479096583692    |
|   opponent2_pre_spike_headshots            |   0.0018462102374419100  |
|   ally4_pre_spike_total_health_loss        |   0.0017627016797871700  |
|   self_pre_spike_total_ammo_mag_loss       |   0.0013313225594919400  |
|   self_pre_spike_avg_loadout_value         |   0.0012452141909295800  |
|   ally3_pre_spike_total_health_loss        |   0.0010148915823726200  |
|   self_pre_spike_longest_gun_primary_none  |   0.0008833539099842530  |
|   self_pre_spike_total_ability_usage_3     |   0.0007852034755415590  |
|   round_info_round_length                  |   0.0007662888826076610  |

Here, we see some differences with spike_time, credits, shield, loadout value.

####  What should player's focus on to win a round of Valorant?: Pre vs Post Spike plant

##### Metrics: Pre-spike, tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 98%      |
| Test Accuracy  | 92%      |
| Train F1       | 98%      |
| Test F1        | 92%      |
##### Metrics: Pre-spike, pruning (max 5 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 89%      |
| Test Accuracy  | 89%      |
| Train F1       | 89%      |
| Test F1        | 89%      |
<img src="dtree_csv_feature_results/confusion_mat_pres.png" alt="" width="400"/>

##### Top 20 important features combined: Pre-Spike
| Feature                                 | Importance                |
|-----------------------------------------|---------------------------|
|   all_opponent_dead                     |   0.5864811183074020      |
|   all_ally_dead                         |   0.23709335028840600     |
|   self_pre_spike_total_health_loss      |   0.05479824356595040     |
|   spike_planted                         |   0.03319837451868730     |
|   ally2_pre_spike_deaths                |   0.02426883741788350     |
|   round_info_ally_side_defender         |   0.02344223769422470     |
|   ally1_pre_spike_deaths                |   0.01818402869285020     |
|   round_info_round_length               |   0.007737994203546460    |
|   self_pre_spike_map_covered            |   0.003619079556952700    |
|   self_pre_spike_max_ammo_reserve_loss  |   0.0021414621094502700   |
|   self_longest_inv_state_melee          |   0.001680049485655450    |
|   ally2_pre_spike_elims                 |   0.001340261105628430    |
|   self_pre_spike_avg_shield             |   0.0011978462311038      |
|   ally3_pre_spike_max_health_loss       |   0.0009076748148840770   |
|   opponent8_character_breach            |   0.000819649753192898    |
|   ally3_pre_spike_avg_health            |   0.0007378068393542700   |
|   self_pre_spike_avg_credits            |   0.000700366369509319    |
|   spike_time                            |   0.0006067974225428740   |
|   self_pre_spike_assists                |   0.000570047872594907    |
|   ally4_pre_spike_deaths                |   0.00043275176800984900  |

We still see deaths/health as big predictors, but we also see spike information, round length information, map covered, ammo, shield, credits.

##### Metrics: Post-spike, tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 98%      |
| Test Accuracy  | 91%      |
| Train F1       | 98%      |
| Test F1        | 91%      |
##### Metrics: Post-spike, pruning (max 5 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 83%      |
| Test Accuracy  | 83%      |
| Train F1       | 83%      |
| Test F1        | 83%      |
<img src="dtree_csv_feature_results/confusion_mat_posts.png" alt="" width="400"/>

##### Top 20 important features combined: Post-Spike
| Feature                                   | Importance               |
|-------------------------------------------|--------------------------|
|   all_opponent_dead                       |   0.6280400593330800     |
|   all_ally_dead                           |   0.2538941444736540     |
|   self_post_spike_avg_credits             |   0.03714718395035830    |
|   self_post_spike_deaths                  |   0.02742440874280730    |
|   round_info_ally_side_attacker           |   0.014867463355890300   |
|   self_post_spike_longest_inv_state_none  |   0.01094221911780830    |
|   opponent2_post_spike_assists            |   0.004798871877568960   |
|   ally1_post_spike_max_health_loss        |   0.00446174679437581    |
|   ally4_post_spike_max_health_loss        |   0.0042124018553596000  |
|   ally2_post_spike_assists                |   0.002468327796974460   |
|   ally2_post_spike_elims                  |   0.0018685143494803800  |
|   ally3_character_reyna                   |   0.0016012022906741300  |
|   ally2_post_spike_headshots              |   0.0012959920374819800  |
|   opponent0_post_spike_elims              |   0.0012158195316938300  |
|   ally4_post_spike_avg_health             |   0.001205967223724770   |
|   self_post_spike_total_health_loss       |   0.0011868985078241100  |
|   opponent2_post_spike_headshots          |   0.0009396836676187030  |
|   round_info_round_length                 |   0.0009174824635377740  |
|   ally1_post_spike_elims                  |   0.0008730893726194370  |
|   user_id                                 |   0.0005531789542340350  |

We still see deaths/health as big predictors, but also attacking side, credits, ability assists, elims.

## Do strategies change depending on what "role" you are playing as?
Lastly, after successfully showing that Valorant players can be classified into 4 roles (Valorant classes) based on gameplay alone, I became curious if this would alter strategy. So, I decided to do a final permutation where I break up my data into the 4 classes and try to predict match outcomes.

I focused on the decision tree model for this question, following the same process as outlined above.

### Results
| Data Sub-Set | Main Selected Features                                            |
|--------------|--------------------------------------------------------------|
| Sentinels          | Deaths, Health, Credits, Eliminations                                         |
| Controllers    | Deaths, Health, Shield, Spike_time                                               |
| Duelists       | Deaths, Health, Eliminations, Ammo 
| Initiators       | Deaths, Health, Eliminations, Ammo |

##### Metrics: Sentinels, tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 99%      |
| Test Accuracy  | 95%      |
| Train F1       | 99%      |
| Test F1        | 95%      |
##### Metrics: Sentinels, pruning (max 4 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 92%      |
| Test Accuracy  | 91%      |
| Train F1       | 92%      |
| Test F1        | 91%      |
<img src="dtree_csv_feature_results/confusion_mat_sentinels.png" alt="" width="400"/>

##### Top 13 important features combined: Sentinels
| Feature                                         | Importance               |
|-------------------------------------------------|--------------------------|
|   all_opponent_dead                             |   0.6466329098541790     |
|   all_ally_dead                                 |   0.16605410217630400    |
|   self_post_spike_total_health_loss             |   0.08102841907642800    |
|   self_pre_spike_avg_health                     |   0.04821851694769530    |
|   ally2_pre_spike_deaths                        |   0.024802251548501400   |
|   ally4_post_spike_total_health_loss            |   0.00914601419145723    |
|   self_post_spike_avg_credits                   |   0.008933718120848090   |
|   ally3_pre_spike_elims                         |   0.005158124225424690   |
|   ally4_pre_spike_deaths                        |   0.003528040915268060   |
|   ally1_pre_spike_total_health_loss             |   0.002553814996320940   |
|   self_post_spike_longest_gun_secondary_shorty  |   0.0018790939417949200  |
|   self_pre_spike_avg_ammo_mag                   |   0.0013899905001127000  |
|   ally1_post_spike_elims                        |   0.0006750035056651190  |

Major Features are: Deaths, Health, Credits, Elims 

##### Metrics: Controllers, tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 99%      |
| Test Accuracy  | 95%      |
| Train F1       | 99%      |
| Test F1        | 95%      |
##### Metrics: Controllers, pruning (max 4 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 88%      |
| Test Accuracy  | 87%      |
| Train F1       | 87%      |
| Test F1        | 86%      |
<img src="dtree_csv_feature_results/confusion_mat_controllers.png" alt="" width="400"/>

##### Top 13 important features combined: Controllers
| Feature                              | Importance               |
|--------------------------------------|--------------------------|
|   all_opponent_dead                  |   0.5313251546593850     |
|   all_ally_dead                      |   0.21802057880589000    |
|   self_post_spike_total_health_loss  |   0.10772456083419200    |
|   self_pre_spike_total_health_loss   |   0.08137508163848770    |
|   ally2_pre_spike_deaths             |   0.023773004750971000   |
|   self_post_spike_avg_shield         |   0.011771360307104600   |
|   spike_time                         |   0.0072919269090518500  |
|   ally2_post_spike_deaths            |   0.004013170041574490   |
|   ally2_pre_spike_avg_health         |   0.003803743087406730   |
|   self_longest_inv_state_melee       |   0.0029526883751678100  |
|   ally2_post_spike_assists           |   0.0027898924856679400  |
|   self_pre_spike_avg_loadout_value   |   0.0022700759831128400  |
|   ally1_pre_spike_total_health_loss  |   0.0020018028213818300  |
|   opponent1_pre_spike_elims          |   0.0008869593006061930  |

Major Features are: Deaths, Health, Shield, Spike_time 

##### Metrics: Duelists, tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 98%      |
| Test Accuracy  | 86%      |
| Train F1       | 98%      |
| Test F1        | 87%      |
##### Metrics: Duelists, pruning (max 4 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 85%      |
| Test Accuracy  | 83%      |
| Train F1       | 86%      |
| Test F1        | 84%      |
<img src="dtree_csv_feature_results/confusion_mat_duelists.png" alt="" width="400"/>

##### Top 13 important features combined: Duelists
| Feature                                | Importance               |
|----------------------------------------|--------------------------|
|   all_ally_dead                        |   0.5558661283941320     |
|   all_opponent_dead                    |   0.2814103617309800     |
|   self_post_spike_total_health_loss    |   0.06440872099054870    |
|   self_pre_spike_total_health_loss     |   0.04763493540957710    |
|   ally3_pre_spike_deaths               |   0.028047249420188600   |
|   ally4_post_spike_total_health_loss   |   0.011280802431473400   |
|   opponent0_pre_spike_assists          |   0.0038790350242521200  |
|   self_post_spike_avg_ammo_mag         |   0.002284337415337090   |
|   opponent2_post_spike_elims           |   0.0018716792821606200  |
|   opponent4_post_spike_ultimate_usage  |   0.0012214041734913800  |
|   opponent1_post_spike_headshots       |   0.0011858881424789400  |
|   opponent4_pre_spike_elims            |   0.0004610072240330830  |
|   round_info_round_start               |   0.0004484503613470860  |
|   opponent1_pre_spike_elims            |   0.0008869593006061930  |

Major Features are: Deaths, Health, Eliminations, Ammo 



##### Metrics: Initiators, tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 98%      |
| Test Accuracy  | 94%      |
| Train F1       | 98%      |
| Test F1        | 94%      |
##### Metrics: Initiators, pruning (max 4 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 90%      |
| Test Accuracy  | 90%      |
| Train F1       | 90%      |
| Test F1        | 90%      |
<img src="dtree_csv_feature_results/confusion_mat_initiators.png" alt="" width="400"/>

##### Top 13 important features combined: Initiators
| Feature                                 | Importance               |
|-----------------------------------------|--------------------------|
|   all_ally_dead                         |   0.5976543649247500     |
|   all_opponent_dead                     |   0.2250098246834780     |
|   self_post_spike_total_health_loss     |   0.06528313766617960    |
|   self_pre_spike_deaths                 |   0.06375183353134430    |
|   opponent1_pre_spike_deaths            |   0.018735085958407500   |
|   self_post_spike_deaths                |   0.007670747512733130   |
|   ally2_post_spike_elims                |   0.007208811202276410   |
|   self_pre_spike_max_ammo_reserve_loss  |   0.006280862687168180   |
|   self_pre_spike_max_ammo_mag_loss      |   0.0038895373923265100  |
|   opponent4_post_spike_elims            |   0.002407547081122420   |
|   round_info_round_end                  |   0.00105894589084887    |
|   ally2_pre_spike_avg_health            |   0.0010493014693654300  |
|   round_info_round_start                |   0.0004484503613470860  |
|   opponent1_pre_spike_elims             |   0.0008869593006061930  |

Major Features are: Deaths, Health, Eliminations, Ammo 