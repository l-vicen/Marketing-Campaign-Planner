# Dependencies
import streamlit as st
from inform import Descriptions

def display_video_page():

    """
    display_video_page() is responsable for 
    displaying all the videos in the video page.
    """

    st.title("Videos")
    st.info(Descriptions.VIDEOS_ABOUT)
    st.markdown('---')

    c1, c2, c3 = st.columns([1, 3, 1])

    # PREPROCESSING VIDEO TUTORIAL
    c2.markdown("## I. Preprocessing Tutorial")
    video_file_prepProp = open('assets/videos/preprocessing.mp4', 'rb')
    video_bytes_prepProp = video_file_prepProp.read()
    c2.video(video_bytes_prepProp)
    c2.markdown('---')

    # CUSTOMER SEGMENTATION VIDEO TUTORIAL
    c2.markdown("## II. States Tutorial")
    video_file_states = open('assets/videos/states.mp4', 'rb')
    video_bytes_states = video_file_states.read()
    c2.video(video_bytes_states)
    c2.markdown('---')

    # CUSTOMER DYNAMICS VIDEO TUTORIAL
    c2.markdown("## III. Transitional Probabilities Tutorial")
    video_file_trans = open('assets/videos/trans.mp4', 'rb')
    video_bytes_trans = video_file_trans.read()
    c2.video(video_bytes_trans)
    c2.markdown('---')

    # REWARDS VIDEO TUTORIAL
    c2.markdown("## III. Rewards Tutorial")
    video_file_rewards = open('assets/videos/rewards.mp4', 'rb')
    video_bytes_rewards = video_file_rewards.read()
    c2.video(video_bytes_rewards)
    c2.markdown('---')

    # MDP VIDEO TUTORIAL
    c2.markdown("## IV. MDP Solver Tutorial")
    video_file_mdp = open('assets/videos/mdp.mp4', 'rb')
    video_bytes_mdp = video_file_mdp.read()
    c2.video(video_bytes_mdp)
    c2.markdown('---')

    # MCP VIDEO TUTORIAL
    c2.markdown("## IV. Marketing Campaign Planner")
    video_file_mcp = open('assets/videos/mcp.mp4', 'rb')
    video_bytes_mcp = video_file_mcp.read()

    c2.video(video_bytes_mcp)
    c2.markdown('---')

    
    
