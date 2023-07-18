## What should player's focus on to win a round of Valorant?
The purpose of this document is to walk through the first batch of research questions related to winning rounds in Valorant.

## Additional context
- To win a match of valorant, you need to win 13 rounds (5 if swiftplay)
    - To win a round, how you win is dependent on if you are attacker or defender
        - If attacker, you win by either planting the spike and it detonates, or you eliminate all opponents
        - If defender, you win by either deativating a planted spike, eliminating all opponents, or just suriving without the spike being planted
- Hypothesis: depending on if you are attacker or defender, and if the spike has been planted or not, your strategy will change
    - First, stay alive
    - If you are an attacker and the spike is not planted, should focus on planting the spike
    - If you are an attacker and the spike is planted, all teammates should focus on defending the spike and killing the other team
    - If you are a defender and the spike is not planted, the team should focus on kills, defending site
    - If you are a defender and the spike is planted, the team should focus on kills/deativating the spike

## Are there certain characters, maps that are more likely to lead to a win?
This is more of an exploratoy question. A good, balanced, game would not have one map that was siginificaly easier to win on, or one character that was dominant over the rest. To assess this, I grouped the data by feature and calculated the win-ratio.
<img src="../imgs/wr_map.png" alt="win ratio map" width="400"/>
<img src="../imgs/wr_agent.png" alt="win ratio agent" width="400"/>
<img src="../imgs/wr_ad.png" alt="win ratio attack defend" width="400"/>
<img src="../imgs/wr_spike.png" alt="win ratio spike" width="400"/>

I found the map types and agents to be pretty balanced. There did not seem to be any one map or agent that was "easier" to win with. I also checked the win/loss ratio for attackers vs defenders, as well as spike planted vs not. Like the map and agent, this seemed to indicate that its not "easier" to be an attacker or defender, or that its always better to plant the spike. 

None of this was suprising, in fact this is what one wants to see when dealing with a "balanced" game. This means that the things players need to do to win a round is much more subtle than simply picking the best agent. 

Refer to the [EDA section](eda.md) for information on the EDA process and how these visuals were generated.

## What should player's focus on to win a round of Valorant?
To answer this wu

### Does this change depending on if you are attacking or defending?

### Does this change depending on what point of the match you are in (pre or post spike plant)?

### Does this change depending on what "role" you are playing as?