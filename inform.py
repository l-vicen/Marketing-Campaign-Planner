"""
All Text Inputs have been defined here.
"""

class Descriptions:

    # HOME PAGE

    ABOUT = "MCP is a python-based app that allows policymakers to plan optimal marketing campaigns using a Markov Decision Process at the customer level. \
        With MCP policymakers are able to segment their customers in different Customer Lifetime Values (CLV), identify the dynamic behavior of its customer based on \
        the likelihood of them noticing a marketing actio, and define the reward in case a customer moves between CLV levels."

    # RESEARCH PAGE

    RESEARCH_ABOUT = 'In this page, users can find the final report that motivated us to develop this application. The report below is the underlying foundation of the app. \
        Furthermore, the app does not reflect all features suggested by the model report as the report calls for high sophistication in implementation and the resources for this project were \
        limited.'

    # VIDEO PAGE

    VIDEOS_ABOUT = 'In this page, users can find a tutorial for each app feature/part. With the videos users can get an idea on how the app has been conceptualized, serving as a guide through \
        users experience.'

    # PREPROCESSING PAGE

    PREPROCESSING_ABOUT = "Here users can clean their proprietary data in real time. \
        The preprocesing includes data visualization, missing values identification, \
        empty observation removal, mean measurement filling and binary categorical variable encoding."

    PREP_INPUT = '__Input:__ Dataframe'
    PREP_OUTPUT = '__Output:__ Clened Dataframe'

    # CUSTOMER SEGMENTATION PAGE
        
    CART_ABOUT = "Here the Classification and Regression Trees (CART) algorithm is applied using Gini's impurity index splitting criterion on the \
        dataset that ressembles industry standard.s The generated code snippet should then be applied locally to classify own data."

    CART_INPUT = "__Input:__ Dataframe with Comparable's CLV & Comparable's RFM, and Dataframe with own RFM"
    CART_OUTPUT = '__Output:__ Code Snippet (Runnable Locally)'

    # CUSTOMER DYNAMICS PAGE

    PROBABILITY_ABOUT = "Here the customer behavior for all Triples (S, A, S') is calculated. The customer behavior stems from the likelihood of him reactiong to a marketing action."

    PROBABILITY_INPUT = '__Input:__ Dataframe with State and Action sets'
    PROBABILITY_OUTPUT = '__Output:__ Dataframe with Transition Probabilities'

    # REWARD PAGE

    REWARD_ABOUT = "Here the reward for all Triples (S, A, S') is calculated. It calculates the Delta CLV between states, automatically incurs action cost and considers a weighting factor."

    REWARD_INPUT = '__Input:__ Dataframe with State and Action sets, Dataframe with Action Costs and Weighting Factor'
    REWARD_OUTPUT = '__Output:__ Dataframe with Rewards'

    # MARKOV DECISION PROCESS PAGE

    MDP_ABOUT = 'A Markov Decision Process is a set of {States, Inputs, Trans. Prob, Rewards and Discount Factor}.\
        In the Setup section the policy maker is responsable for inputing these values.'

    MDP_INPUT = '__Input:__ Dataframe with State and Action sets, Discounting Factor, Number of Decision Periods, MDP Solver'
    MDP_OUTPUT = '__Output:__ Dataframe Optimal Policy, Optimal Value Function, Performance Algorithm Indicators'

    SOLVERS = 'Different MDP solvers yield different results. Pick your solver, e.g. Value Iteration.'

    # MARKETING CAMPAIGN PLANNER PAGE

    CAMPAIGN_PLANNER_ABOUT = 'In this section, a plan of the average optimal campaign over N simulations is computed.'
    CAMPAIGN_PLANNER_INPUT = '__Input:__ Action Cost, Initial State, Optimal Policy, N simulations'
    CAMPAIGN_PLANNER_OUTPUT = '__Output:__  N optimal campaigns, Total Cost, Total CLV Change, Action Sequence'

    # SIMULATION HISTORY

    SIMULATION_LOG_HISTORY = 'Here users can find the history of all marketing campaigns ever run using MCP. One optimal marketing campaign and its respective Key Performance Indicators (KPIs) is one row of the summary table below.'

descriptions = Descriptions()