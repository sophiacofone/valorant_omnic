# Winning Rounds in Valorant
The purpose of this document is to walk through the first batch of research questions related to winning rounds in Valorant.

## Questions and additional context
- To win a match of valorant, you need to win 13 rounds (5 rounds if the game-type is swiftplay).
    - There are several ways to win a round. The options are dependent on if you are attacking or defending. 
        - If you are on the attacking side, you win by either planting the spike on the opposing side and it detonates, or you eliminate all opponents.
        - If you are on the defending side, you win by either deativating a planted spike, eliminating all opponents, or just suriving without the spike being planted.
- Hypothesis: depending on if you are attacker or defender, and if the spike has been planted or not, your strategy will change
    - If you are an attacker and the spike is not planted, should focus on planting the spike.
    - If you are an attacker and the spike is planted, all teammates should focus on defending the spike and killing the other team.
    - If you are a defender and the spike is not planted, the team should focus on kills and defending the site.
    - If you are a defender and the spike is planted, the team should focus on kills/deativating the spike.

## Are there certain characters, maps that are more likely to lead to a win?
This is more of an EDA question. A well-designed game would not have one map that was siginificaly easier to win on, or one character that was dominant over the rest. To assess this, I grouped the data by feature and calculated the win-ratio.

<img src="../imgs/wr_map.png" alt="win ratio map" width="400"/> <img src="../imgs/wr_agent.png" alt="win ratio agent" width="400"/>
<img src="../imgs/wr_ad.png" alt="win ratio attack defend" width="400"/> <img src="../imgs/wr_spike.png" alt="win ratio spike" width="400"/>

I found the map types and agents to be balanced. Accodrding to this data, there does not seem to be any one map or agent that was "easier" to win with. I also checked the win/loss ratio for attackers vs defenders, as well as spike planted vs not. Like map and agent, these visuals indicate that it is not "easier" to be an attacker or defender, or that its always "better" to plant the spike rather than just eliminating the whole opposing team. 

This is not a suprising result. In fact, this indicates that what it takes to win a round of Valorant is more strategic and subtle than simply picking the best agent (for example). Individual players may still have maps or agents that consistantly give them better outcomes. However, this is likely due to preference/practice rather than the agent/map itself being inherently better.

Refer to the [EDA section](https://github.com/sophiacofone/omnic_ml/blob/main/EDA/eda.md) for more information on the EDA process and how these visuals were generated.

## What should players focus on to win a round of Valorant?
To answer this question, one strategy is to build a classifier that can accuratley predict wins and losses. Then, we can examine the feature importances/coefficents and determine which factors are most influential in determining match outcomes. Since we care less about the actual prediction and more about the features, a good choice would be to work with with models like decision trees and logistic regression since these models can be easily interpreted.

### Data & Preprocessing
Refer to the [preprocess section](https://github.com/sophiacofone/omnic_ml/edit/main/preprocess/preprocess.md) for information on overall data preprocessing. In addition to these steps, `win_loss/win_loss_data_preprocessing.py` drops some irrelevant columns (`['player','round_number']`), re-maps the true/false strings to 1 and 0, and one-hot encodes the other categorial features (`'map','self_character','ally4_character','ally1_character','ally2_character','ally3_character','opponent5_character', 'opponent6_character','opponent7_character','opponent8_character','opponent9_character','round_info_ally_side','self_longest_inv_state','self_longest_gun_primary','self_longest_gun_secondary','self_post_spike_longest_inv_state','self_pre_spike_longest_inv_state','self_post_spike_longest_gun_secondary','self_pre_spike_longest_gun_secondary','self_post_spike_longest_gun_primary','self_pre_spike_longest_gun_primary']`). Finally, I added a feature to capture round length (`['round_info_round_length']`). `win_loss/win_loss_data_preprocessing.py`outputs a csv per dataset, and a combined csv for all the datasets. These csvs are the input data for the modeling below. The combined dataframe has 28959 rows × 542 columns.

## Logistic Regression Model
Logistic regression is usually a good starting point for binary classification problems. As mentioned above, its primary "pro" is that it is simple and easy to interpret. It is also fast, easy to implement, and isn't really prone to overfititng. However, its main downside is that it can be **too** simple. It assumes the data has a linear relationship with the outcome. Therefore, it works best when the data is linearly seperable. Logistic regression struggles to capture complex relationships due to this linealirty assumpotion.

The "goal" of logstic regression is to predict one of two outcomes based on the input features. The logistic function (sigmoid function) maps any number to a value between 0 and 1 (good for probabilities). 

Logistic function: -σ(z) = 1 / (1 + e^(-z))
- σ(z) is the output (probability) between 0 and 1.
- e is the base of the natural logarithm, approximately equal to 2.71828.
- z is the linear combination of the input features and their associated weights.
    - z = w0 * x0 + w1 * x1 + w2 * x2 + ... + wn * xn
    - w0, w1, w2, ..., wn are the coefficients (weights) associated with each feature.
    - x0, x1, x2, ..., xn are the corresponding feature values.

During training, the model finds the best values for the coefficients (weights) that minimize the error between the predicted probabilities and the actual class labels in the training data.

### Class Imbalance
Log reg is sensitive to class inbalance. My first step was to ensure my target was balanced.

<img src="../imgs/g_won.png" alt="win ratio map" width="400"/>

### Feature Selection
Since we are ultimatly interseted in the **features** of this model, I decided to incoperate feature selection into this process. I first trained a logistic regression model using L1 regularizatoin. L1 regularization has the benefit of driving some feature weights to zero, effectivly excluding them from the model. I decided to further eliminate features by only including features with coefs that were above the median value. Then, I used those features to create a new model using L2 regualrization. L2 regularization prevents overfitting and is less sensitive to outliers than L1. In both cases, I used cross validation to tune the "C" hyper parameter (controls the strength of the regularization). 

### Results
####  What should player's focus on to win a round of Valorant?: All data
I created a dataframe/csv (`win_loss/df_coefs_logreg_all.csv`) of the features and their coefs. The magnitude indicates the "importance" of the feature.

##### Metrics
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 95%      |
| Test Accuracy  | 95%      |
| Train F1       | 95%      |
| Test F1        | 95%      |
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

These features indicate that from an overall perspective, your team not dying, your opponents dying, and health are the most important predicter for wining rounds of valorant. This may seem obvious, but in Valorant there are multiple ways to win with elims/deaths being only one of them. These results could justify playing more "defensivly", i.e. not dying versus trying to get lots of elims by taking risky moves.

#### No deaths?
I wanted to also see what without using deaths as a feature, could the model still predict well and what features it would pick.

##### Metrics
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 94%      |
| Test Accuracy  | 94%      |
| Train F1       | 95%      |
| Test F1        | 94%      |
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
Now that we have an idea of what it takes to win a round of valorant from an overall level, I thought it would be interesting to explore if these features change depending on some addiitonal criteria. I decided to use a "stratified analysis" approach, where I devide my data into groups and seperatly preform my analysis and investigate the differences (if there are any).

First, I devided my data into "attacker" and "defender" (I removed the unknown rows).

<img src="../imgs/g_ally.png" alt="win ratio map" width="400"/>

Then, I re-ran my process using `win_loss/csv/'wl_alldf_prepro_data_attack.csv` and `win_loss/csv/'wl_alldf_prepro_data_defend.csv` producing `win_loss/df_coefs_logreg_attack.csv` and `win_loss/df_coefs_logreg_defenc.csv`.

##### Metrics
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 97%      |
| Test Accuracy  | 96%      |
| Train F1       | 97%      |
| Test F1        | 96%      |
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
| Test F1        | 95%      |
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

Here, we see some differences. Deaths/health is still the highest, but we starat to see ammo, gun usage, ability usage, and elims. 

####  What should player's focus on to win a round of Valorant?: Pre vs Post Spike plant

##### Metrics
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 91%      |
| Test Accuracy  | 91%      |
| Train F1       | 91%      |
| Test F1        | 91%      |
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
| Train F1       | 89%      |
| Test F1        | 89%      |
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
Even though I acheied good results with the simple logistic regression models, I wanted to also explore a non-linear classifier like decision trees. Decicion trees work by reecursively partitioning the input space into smaller regions, and making predictions based on the majority class or average value of the target variable in that region. The decision tree starts by selecting the best feature to split the data. The "best" is defiend by the feature that maximizes the seperation of the classes. After the data is split, each subset becomes a new "node" in the tree. This process continues until a stopping criteria it met. Once the tree is finsihed building, we can then assing labels to the termonal notes. 

Again, since we are most interested in the features used for the classification, this model is a good choice. Decision trees are prone to overfitting, but we can overcome that issue via pruning. Decision trees also handle mixed variabale types (categorial and continous) well, which is ideal for our problem space.

Math for splitting?

Hyperparam tuning

### Results
####  What should player's focus on to win a round of Valorant?: All data
I created a dataframe/csv (`win_loss/df_import_dtree_all_5.csv`) of the features and their importances. The magnitude indicates the "importance" of the feature.

##### Metrics: Tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 99%      |
| Test Accuracy  | 93%      |
| Train F1       | 100%      |
| Test F1        | 94%      |
##### Metrics: After pruning (max 7 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 93%      |
| Test Accuracy  | 93%      |
| Train F1       | 93%      |
| Test F1        | 93%      |
##### Metrics: More pruning (max 5 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 90%      |
| Test Accuracy  | 90%      |
| Train F1       | 90%      |
| Test F1        | 90%      |
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

Just like the previous model, these features indicate that from an overall perspective your team not dying, your opponents dying, and health are the most important predicters for wining rounds of valorant. After those features, the model starts using credits, % map covered, assists, and elims. 

#### No deaths?
I wanted to also see what without using deaths as a feature, could the model still predict well and what features it would pick.

##### Metrics: Tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 96%      |
| Test Accuracy  | 90%      |
| Train F1       | 97%      |
| Test F1        | 89%      |
##### Metrics: More pruning (max 5 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 75%      |
| Test Accuracy  | 75%      |
| Train F1       | 74%      |
| Test F1        | 74%      |

self_post_spike_total_health_loss	0.4784037937876600
self_pre_spike_total_health_loss	0.437403878680923
self_post_spike_avg_credits	0.08419232753141710

##### Top 20 important features combined - no deaths
|   Feature                                 |   Importance           |
|-------------------------------------------|------------------------|
|   self_post_spike_total_health_loss       |   0.4784037937876600   |
|   self_pre_spike_total_health_loss        |   0.437403878680923    |
|   self_post_spike_avg_credits             |   0.08419232753141710  |

This model does not predict as well, espically after pruning. It seems like this model is more sensitive to the death-related information. 

####  What should player's focus on to win a round of Valorant?: Attacking vs Defending
Here, I stick with the same "stratified analysis" approach.

##### Metrics: Attack, tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 98%      |
| Test Accuracy  | 95%      |
| Train F1       | 99%      |
| Test F1        | 94%      |
##### Metrics: Attack, pruning (max 5 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 88%      |
| Test Accuracy  | 88%      |
| Train F1       | 88%      |
| Test F1        | 88%      |
##### Top 20 important features combined: Attack
| Feature                               | Importance               |
|---------------------------------------|--------------------------|
|   all_ally_dead                       |   0.5874870837653580     |
|   all_opponent_dead                   |   0.2629095784315320     |
|   self_pre_spike_total_health_loss    |   0.13122116652423000    |
|   ally4_post_spike_total_health_loss  |   0.00927680739040505    |
|   ally1_pre_spike_elims               |   0.007971353122552120   |
|   ally3_pre_spike_avg_health          |   0.0011340107659231400  |

Similar to the analysis above, we see deaths/not dying/health as the best thing to focus on. However, we do see some elim features here. 

##### Metrics: Defend, tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 98%      |
| Test Accuracy  | 93%      |
| Train F1       | 98%      |
| Test F1        | 93%      |
##### Metrics: Defend, pruning (max 5 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 87%      |
| Test Accuracy  | 87%      |
| Train F1       | 88%      |
| Test F1        | 88%      |
##### Top 20 important features combined: Defend
| Feature                             | Importance               |
|-------------------------------------|--------------------------|
|   all_ally_dead                     |   0.6205993085790630     |
|   all_opponent_dead                 |   0.25486827994733600    |
|   self_pre_spike_total_health_loss  |   0.10701466156996800    |
|   ally4_post_spike_max_health_loss  |   0.012061561551254200   |
|   spike_time                        |   0.004569102542299530   |
|   round_info_round_length           |   0.0008870858100790660  |

Here, we see some differences with spike_time and round_length.

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
| Train Accuracy | 85%      |
| Test Accuracy  | 85%      |
| Train F1       | 85%      |
| Test F1        | 85%      |
##### Top 20 important features combined: Pre-Spike
| Feature                                 | Importance               |
|-----------------------------------------|--------------------------|
|   all_opponent_dead                     |   0.6638165862301630     |
|   all_ally_dead                         |   0.2683573153395690     |
|   self_pre_spike_total_health_loss      |   0.06202413315596620    |
|   round_info_round_length               |   0.002543025770929630   |
|   self_pre_spike_max_ammo_reserve_loss  |   0.0024238428530130900  |
|   ally3_pre_spike_avg_health            |   0.0008350966503591820  |

We still see deaths/health as big predictors, but with more gun-related features.

##### Metrics: Post-spike, tuned, no pruning
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 98%      |
| Test Accuracy  | 92%      |
| Train F1       | 91%      |
| Test F1        | 91%      |
##### Metrics: Post-spike, pruning (max 5 depth)
| Metric         | Result   |
| -------------- | -------- |
| Train Accuracy | 85%      |
| Test Accuracy  | 85%      |
| Train F1       | 85%      |
| Test F1        | 85%      |
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

### Does this change depending on what "role" you are playing as?