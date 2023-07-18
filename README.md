# Gameplay-Based Classification of Valorant Players: Insights and Feature Importance
This repo documents the development efforts in investigating Valorant gameplay data using data analysis, statistical methods, and machine learning.

## Motivation and Research Questions
The motivation behind this project is to try and prove (or disprove) common sentiments regarding Valorant gameplay strategy. In the Valorant gaming community, it is common for certain play-styles, characters, weapons, and strategies to be considered "meta" (aka very strong or powerful). By following these strategies, picking certain characters, or utilizing specific weapons, players should have an edge over the competition and garner better match outcomes. However, there is very little actual data (or analysis) to support these claims (it is mainly anecdotal accounts). 

My goal is to formulate some of these claims into research questions, and use machine learning & data analysis to investigate them. The main questions I considered for this analysis are:

1. What should player's focus on to win a round of Valorant?
   - Are there certain characters, weapons, maps that are more likely to lead to a win?
   - Does this change depending on if you are attacking or defending?
   - Does this change depending on what point of the match you are in (pre or post spike plant)?
   - Does this change depending on what "role" you are playing as?
2. Is it possible to classify Valorant players into "roles" based solely on game-play data?
   - Do sentinels, controllers, duelists, and initiators truly use different strategies and gameplay mechanics?
   - Are there other roles that are more informative of how users actually play?
     - For example, in the Valorant community a commonly-recommended strategy is to play as a "entry-fragger", which is a more specific role than than just "duelist".
    
Much more detail into the background & context behind these questions is provided in the [win_condition](https://github.com/sophiacofone/omnic_ml/tree/main/notebooks) and [role](https://github.com/sophiacofone/omnic_ml/tree/main/notebooks) sections. 

## About the Data
The data for this project is provided by [Omnic Data ](https://www.omnic.ai/forge.html). Omnic Data is an AI based E-Sports coaching platform that uses computer vision to collect data from uploaded videos of E-sports matches. Omnic provided ~X rounds of Valorant gameplay from a verity of professional Valorant streamers for this project. One of the main advantages of this dataset is it only includes professional players, hence we can expect less "noise" when compared to all user data. The data is a time-series, where each event is captured with a timestamp.

## Repository Overview
The repo is organized into three main sections. The [notebooks](https://github.com/sophiacofone/omnic_ml/tree/main/notebooks) section contains all the notebooks used

## Main Results

## Challenges
The main challenges for this project were data pre-processing, cleaning, feature engineering, lack of "related work", and the "limited perspective" of Omnic's data.  

**Data pre-processing & cleaning:** The data for this project was provided as raw time-series JSON. The data needed to be cleaned (eliminate redundancies and inconstancies) and parsed into a flat format usable for modeling.

**Feature engineering:** Using domain expertise (myself and the folks at Omnic data), I was able to create a host of new features that were derived from the raw data, but helped capture the gameplay events more accurately.

**Lack of related work:** There is very little publicly available data in the E-Sports space. Hence, there isn't a "baseline" that I can use to demonstrate the quality of my models. I drew inspiration from traditional sports analytics (football, basktball, etc) and [Winning duels in VALORANT, a visualization of optimal positioning](https://global-uploads.webflow.com/5f1af76ed86d6771ad48324b/6228f96dd382261a4887643f_Winning%20Duels%20in%20Valorant.pdf) from the SSAC 2022 conference.

**Limited perspective:** Due to the nature of Omnic's data collection process, the data from each round of Valorant is from the perspective of the "active" player. The active player can see some information about teammates and opponents (like opponent eliminations, and teammate health), but the vast majority of the information is limited to only the active player. Given that Valorant is a cooperative game, this is an unfortunate limitation. I believe that once Omnic does have team-based data, the models I have built would preform even better (in terms of classification accuracy and cluster separability) with this additional information. 

## Acknowledgments
As previously mentioned, the data for this academic project was provided by Omnic Data. The Omnic team also provided domain expertise and guidance. This project was also supported by Professor Phillip Bogden and Northeastern University, Roux Institute.
