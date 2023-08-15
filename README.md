# Gameplay-Based Classification of Valorant Players: Insights and Feature Importance
This repo documents the development efforts in investigating Valorant gameplay data using machine learning and data analysis.

## Motivation and Research Questions
The motivation behind this project is to investigate Valorant gameplay strategy. In the Valorant gaming community, it is common for certain play-styles, characters, weapons, and strategies to be considered "meta" (aka very strong or powerful). By following these strategies, picking certain characters, or utilizing specific weapons, players should have an edge over the competition and garner better match outcomes. However, there is little publicly available actual data/analysis to support these claims (most strategies are communicated through anecdotal accounts).

My goal is to formulate some of these claims into research questions, and use machine learning & data analysis to investigate them. The main questions I considered for this analysis are:

1. What should players focus on to win a round of Valorant?
   - Are there certain characters, maps that are more likely to lead to a win?
   - Should your focus change depending on if you are playing as the attecker vs defender?
   - Should your focus change depending on what point of the match you are in (pre or post spike plant)?
   - Should your focus change depending on what "class" you are playing as?
2. Is it possible to classify Valorant players into "classes" based solely on game-play data?
   - Do sentinels, controllers, duelists, and initiators truly use different strategies and gameplay mechanics?
   - Are there other classes/roles that are more informative of how users actually play?
      - For example, in the Valorant community, a commonly-recommended strategy is to play as a "entry-fragger", which is a more specific role than than just "duelist".
    
Much more detail into the background & context behind these questions is provided in the [win_condition](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/win_loss.md) and [role](https://github.com/sophiacofone/omnic_ml/blob/main/roles/roles.md) sections. 

## About the Data
The data for this project is provided by [Omnic Data ](https://www.omnic.ai/forge.html). Omnic Data is an AI based E-Sports coaching platform that uses computer vision to collect data from uploaded videos of E-sports matches. Omnic provided ~30k rounds of Valorant gameplay from professional Valorant streamers for this project. One of the main advantages of this dataset is it only includes professional players, hence we can expect less "noise" when compared to all user data. The data is organized as a time-series, where each event is captured with a timestamp.

## Repository Overview
The repo is organized into 2 main sections. The [win_condition](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/win_loss.md) section handles the first batch of questions regarding what gameplay aspects players should focus on to win rounds of Valorant. The [role](https://github.com/sophiacofone/omnic_ml/blob/main/roles/roles.md) section focuses on using gameplay data to classify players into the 4 Valorant classes.

This repo also includes [preprocess](https://github.com/sophiacofone/omnic_ml/blob/main/preprocess/preprocess.md) which details the extensive data [parsing](https://github.com/sophiacofone/omnic_ml/tree/main/parsing), cleaning, and feature engineering process. 

## Main Results
### What should players focus on to win a round of Valorant?
Firstly, from my EDA process I quickly realized that there was no one map or agent that was associated with better match outcomes (see [win_condition](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/win_loss.md) for details). This is not a surprising result. In fact, this indicates that what it takes to win a round of Valorant is more strategic and subtle than simply picking the best agent (for example).

Then, I explored using both logistic regression and decision trees to try and classify rounds of Valorant into winning rounds and losing rounds. I looked at the features used in the model to make an assessment of what features determined match outcomes. I explored various permutations, such as at attacking vs defending, pre vs post spike.

These models were generally quite accurate, with test accuracy and F1 scores from 90%-95% depending on the permutation (see [win_condition](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/win_loss.md) for details). The summary of the feature importances are as follows:

#### Logistic Regression
<img src="logreg_csv_feature_results/feat_vis_all.png" alt="" width="800"/>
<img src="logreg_csv_feature_results/feat_vis_no_deaths.png" alt="" width="800"/>

<img src="logreg_csv_feature_results/feat_vis_attack.png" alt="" width="800"/>
<img src="logreg_csv_feature_results/feat_vis_defend.png" alt="" width="800"/>

<img src="logreg_csv_feature_results/feat_vis_prespike.png" alt="" width="800"/>
<img src="logreg_csv_feature_results/feat_vis_postspike.png" alt="" width="800"/>

#### Decision tree
<img src="dtree_csv_feature_results/feat_vis_all_5.png" alt="" width="800"/>
<img src="dtree_csv_feature_results/feat_vis_no_deaths_5.png" alt="" width="800"/>

<img src="dtree_csv_feature_results/feat_vis_attack_5.png" alt="" width="800"/>
<img src="dtree_csv_feature_results/feat_vis_defend_5.png" alt="" width="800"/>

<img src="dtree_csv_feature_results/feat_vis_pres_5.png" alt="" width="800"/>
<img src="dtree_csv_feature_results/feat_vis_posts_5.png" alt="" width="800"/>

Both models relied on death and health information no matter the permutation. These features indicate that from an overall perspective, your team not dying, your opponents dying, and health are the most important predictors for winning rounds of Valorant. This may seem obvious, but in Valorant there are multiple ways to win with elims/deaths being only one of them. These results could justify playing more "defensively", i.e. not dying versus trying to get lots of elims by taking risky moves.

If death information is removed, models then relied on elimination based features.

The other permutations offered some variations, in particular for post-spike and defense.

In summary, the two models performed predominantly the same, although some of the finer details differed. I think this analysis could be useful if a player seems to be struggling with a particular type of round. For example, if a user is consistently failing on defense (in addition to prioritizing staying alive and health), a suggestion could be to keep a close watch on the spike and go into defense rounds with strong economy (credits).

Based on these results, I do think that the data shows that the high-level objective stays the same, but the finer-points of strategy does change depending on what's happening in the round (spike) and defense vs attack. 

### Is it possible to classify Valorant players into "classes" based solely on game-play data?

For this question, I stuck with the decision tree model for the same reasons as the previous analysis (we care about the features, needs to be interpretable). Initially, running this model with all the data I had created highly accurate results (+95%). However, I noticed that my model used a lot of features that were essentially the players' "choice" (like user_id, the agent types of teammates, and the map). My interpretation of these initial results is that users have a preference for a class, and then select their class in relation to their teammates and the type of map. This makes sense, as it is common knowledge that certain classes are better for certain maps, and that it is good to have a "variety" of classes on a team.


Once these features were removed, I was left with solely game-play based features. My accuracy and F1 score for this model is 82%. Therefore, I think it is possible to classify players into classes based on this limited feature space (see [roles](https://github.com/sophiacofone/omnic_ml/blob/main/roles/roles.md) for details).

### Are there other classes/roles that are more informative of how users actually play?
As I described in the Motivation section above, the Valorant community believes that the "true" positions/roles in the game are more specific than the predefined Valorant classes. After hypothesizing what these roles might be and doing strategic feature engineering to try and provide as much of that information to my models as I could, I decided to try and tackle this problem via unsupervised clustering (see [roles](https://github.com/sophiacofone/omnic_ml/blob/main/roles/roles.md) for details).

Unfortunately, this ended up being challenging. Despite having classification success earlier in the project, the data was not forming good clusters with the methods I tried. I tried using PCA, Kernel PCA, and T-SNE. None of these methods created easily definable clusters. 

### Should your focus change depending on what "class" you are playing as?
After I was successfully able to classify players into the original Valorant classes, I became curious if a players' class would impact strategy. Similar to the win_loss classification process, I subdivide my data and created a decision tree model (see [win_condition](https://github.com/sophiacofone/omnic_ml/blob/main/win_loss/win_loss.md) for details).

<img src="logreg_csv_feature_results/feat_vis_sentinels.png" alt="" width="800"/>
<img src="logreg_csv_feature_results/feat_vis_controllers.png" alt="" width="800"/>
<img src="logreg_csv_feature_results/feat_vis_duelists.png" alt="" width="800"/>
<img src="logreg_csv_feature_results/feat_vis_initiators.png" alt="" width="800"/>

Like the previous section, deaths and health reigned supreme as features to focus on. However, I was surprised there wasn't more variation here. For example, I expected duelists to rely much more on eliminations. It does make sense that initiators would be the most variable, as they are a very flexible role. I am curious how more team information could change/influence this analysis. 

## Challenges / Limitations

**Lack of related work:** There is very little publicly available data in the E-Sports space. Hence, there isn't a "baseline" that I can use to demonstrate the quality of my models. I drew inspiration from traditional sports analytics (football, basketball, etc.) and [Winning duels in VALORANT, a visualization of optimal positioning](https://global-uploads.webflow.com/5f1af76ed86d6771ad48324b/6228f96dd382261a4887643f_Winning%20Duels%20in%20Valorant.pdf) from the SSAC 2022 conference.

**Limited perspective:** Due to the nature of Omnic's data collection process, the data from each round of Valorant is from the perspective of the "active" player. The active player can see some information about teammates and opponents (like opponent eliminations, and teammate health), but the vast majority of the information is limited to only the active player. Given that Valorant is a cooperative game, this is an unfortunate limitation. The paper I referenced above uses data directly from Valorant's API, therefore the analysis is able to have a more comprehensive view of the game and the interactions between players.

It is ultimately unknown how team-based data would impact my models/results. However, I hypothesize that my clustering idea may have worked better with a complete information space (or at least full teammate data). This incomplete perspective could also be why the win/loss classifier liked elimns, health, and death features so much (these are features that I had "more" information for).

**Sparse Data**: Since I had a mix of categorical and numeric features, the data became some-what sparse after pre-processing. I think this likely had an impact on the unsupervised method, since clustering algorithms usually do not do well with sparse data. 

## Future work
1. Once team-based data becomes available, I would like to try my unsupervised clustering idea again to see if this additional data would help.
2. Labeling! Another idea could be to transform the unsupervised problem into supervised. This is obviously time consuming (and thus out-of-scope for this project), but the possible model choices increase with supervised learning.
3. I would also like to try and identify important "events" that lead up to a win or loss. For this analysis, the parsing/preprocessing step was difficult and time consuming. However, given more time, I would further explore how to parse time-series data and create a more detailed analysis.

## Acknowledgments
As previously mentioned, the data for this academic project was provided by Omnic Data. The Omnic team also provided domain expertise and guidance. This project was also supported by Professor Phillip Bogden and Northeastern University, Roux Institute.