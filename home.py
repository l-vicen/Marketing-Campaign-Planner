# Dependencies
import streamlit as st
import pandas as pd
from inform import Descriptions

def display_home():

    """ Home Page """

    st.title('Marketing Campaign Planner (MCP)')
    st.write('---')

    col1, col2 = st.columns([1,1])
    col2.header('Description')
    col2.info(Descriptions.ABOUT)

    col1.header('App Walkthrough')
    video_file_walk = open('assets/videos/intro.mp4', 'rb')
    video_bytes_walk = video_file_walk.read()

    col1.video(video_bytes_walk)
    st.markdown('---')

class Sidebar: 

    """ Sidebar """

    # Sidebar attribute Logo
    def sidebar_functionality(self):
        st.sidebar.image('assets/tum.png')
        st.sidebar.markdown('---')

    # Authors Section
    def sidebar_contact(self):
        st.sidebar.markdown('##### Contributors')
        st.sidebar.markdown('Julius Miers')
        st.sidebar.markdown('Lucas Perasolo')
        st.sidebar.markdown('---')

sidebar = Sidebar()