import pandas as pd
import streamlit as st

# Drop all observations which Column Value is NaN
def drop(data, columns_list):
    return data.dropna(subset = columns_list)

# Fill any missing data with the mean of observations
def fill(data):
    return data.fillna(data.mean())

# Encode categorical columns and store enconding in flag columns
def createFlag(data, columns, flags):
    for i in range (len(columns)):
        data[flags[i]] = data[columns[i]].apply(lambda x: 1 if x=='Yes' else 0)
    return data
