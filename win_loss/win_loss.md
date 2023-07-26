# Winning Rounds in Valorant
The purpose of this document is to walk through the first batch of research questions related to winning rounds in Valorant.

## Questions and additional context
- To win a match of valorant, you need to win 13 rounds (5 rounds if the gametype is swiftplay)
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

<img src="../imgs/wr_map.png" alt="win ratio map" width="400"/>
<img src="../imgs/wr_agent.png" alt="win ratio agent" width="400"/>
<img src="../imgs/wr_ad.png" alt="win ratio attack defend" width="400"/>
<img src="../imgs/wr_spike.png" alt="win ratio spike" width="400"/>

I found the map types and agents to be balanced. Accodrding to this data, there does not seem to be any one map or agent that was "easier" to win with. I also checked the win/loss ratio for attackers vs defenders, as well as spike planted vs not. Like the map and agent, these visuals indicate that it is not "easier" to be an attacker or defender, or that its always "better" to plant the spike rather than just eliminating the whole opposing team. 

This is not a suprising result. In fact, this indicates that what it takes to win a round of Valorant is more strategic and subtle than simply picking the best agent (for example). Individual players may still have maps or agents that consistantly give them better outcomes. However, this is likely due to preference/practice rather than the agent/map itself being inherently better.

Refer to the [EDA section](https://github.com/sophiacofone/omnic_ml/blob/main/EDA/eda.md) for more information on the EDA process and how these visuals were generated.

## What should player's focus on to win a round of Valorant?
To answer this question, one strategy is to build a classifier that can accuratley predict wins and losses. Then, we can examine the feature importances/coefficents and determine which factors are most influential in determining match outcomes. Since we care less about the actual prediction and more about the features, we need to work with models like decision trees and logistic regression since these models can be easily interpreted.

### Logistic Regression
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

#### Applying Log Reg to this problem
##### Pre-Processing
##### Feature Selection



### Decision Tree
Decision trees work by ...
#### Permutation Importance
Directly measures variable importance by observing the effect on model accuracy of randomly shuffling each predictor variable. It's a more robust way than merely looking at feature importance, especially when features are correlated.

#### Partial Dependence Plots (PDPs
Plots that show how a feature affects predictions. They could help to see whether the relationship between the target and a feature is linear, monotonic, or more complex.


## Does this change depending on if you are attacking or defending?

### Does this change depending on what point of the match you are in (pre or post spike plant)?

### Does this change depending on what "role" you are playing as?