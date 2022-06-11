# Dependencies
import streamlit as st
import pandas as pd
from gsheetsdb import connect
from inform import Descriptions
import gspread

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR

def display_simulation_history():

    """display_simulation_history() is responsable for connecting the app with the database and displaying the simulation history."""

    st.title("Simulations History")
    st.info(Descriptions.SIMULATION_LOG_HISTORY)
    st.markdown('---')

    gsheet_url = "https://docs.google.com/spreadsheets/d/1CHUFijH2220hZfdLDhl6EHsWtSWwlz2BrJi7FkgF44w/edit?usp=sharing"
    conn = connect()
    rows = conn.execute(f'SELECT * FROM "{gsheet_url}"')
    df_gsheet = pd.DataFrame(rows)

    st.markdown('## Results')

    # st.dataframe(df_gsheet.rename(columns={"Simulations_Number": "Simulations Number",
    #                 "Initial_State": "Initial State",
    #                 "No_Contact": "No Contact",
    #                 "Cost_Overall_Best_Action": " Cost Overall Best Action",
    #                 "Average_Clv_Change": "Average Clv Change",
    #                 "Total_Cost_of_Overall_Best_Campaign": "Total Cost of Overall Best Campaign"}), width = 9000, height = 5000)
    st.table(df_gsheet.rename(columns={"Simulations_Number": "Simulations Number",
                    "Initial_State": "Initial State",
                    "No_Contact": "No Contact",
                    "Cost_Overall_Best_Action": " Cost Overall Best Action",
                    "Average_Clv_Change": "Average Clv Change",
                    "Total_Cost_of_Overall_Best_Campaign": "Total Cost of Overall Best Campaign"},))

    st.markdown('---')
    st.markdown('## Visualizations')
    c1, c2 = st.columns(2)
    fig = px.scatter(
        df_gsheet, x='Initial_State', y='Average_Clv_Change', opacity=0.65,
        trendline='ols', trendline_color_override='red', title="Relationship: Initial State & Average CLV Change"
        )
    c1.plotly_chart(fig)

    figTwo = px.scatter(
        df_gsheet, x='Initial_State', y='Total_Cost_of_Overall_Best_Campaign', opacity=0.65,
        trendline='ols', trendline_color_override='red', title="Relationship: Initial State & Total Cost of Overall Best Campaign"
        )
    c2.plotly_chart(figTwo)

    c3, c4 = st.columns(2)
    figThree = px.scatter(
    df_gsheet, x='Simulations_Number', y='Average_Clv_Change', opacity=0.65,
    trendline='ols', trendline_color_override='red', title="Relationship: Number of Simulations & Average CLV Change"
    )
    c3.plotly_chart(figThree)

    figFour = px.scatter(
        df_gsheet, x='Simulations_Number', y='Total_Cost_of_Overall_Best_Campaign', opacity=0.65,
        trendline='ols', trendline_color_override='red', title="Relationship: Number of Simulations & Total Cost of Overall Best Campaign"
        )
    c4.plotly_chart(figFour)

# Save Simulation MCP
def save_simulation(simulations, initial_state,	agent_average,	call_average, email_average, mail_average, no_contact_average, tv_average, cost_overall_best_action, average_clv_change, total_cost_of_overall_best_campaign):

    """
    save_simulation(...) is responsable for updating the simulation history.
    """
    sa = gspread.service_account("credentials.json")
    sh = sa.open("MCP")
    worksheet = sh.get_worksheet(0)

    l = len(worksheet.col_values(1))+1

    worksheet.update_cell(l, 1, simulations)
    worksheet.update_cell(l, 2, initial_state)
    worksheet.update_cell(l, 3, agent_average)
    worksheet.update_cell(l, 4, call_average)
    worksheet.update_cell(l, 5, email_average)
    worksheet.update_cell(l, 6, mail_average)
    worksheet.update_cell(l, 7, no_contact_average)
    worksheet.update_cell(l, 8, tv_average)
    worksheet.update_cell(l, 9, cost_overall_best_action)
    worksheet.update_cell(l, 10, average_clv_change)
    worksheet.update_cell(l, 11,total_cost_of_overall_best_campaign)