# Dependencies
import streamlit as st
import base64
from inform import Descriptions

def display_research_page():

    """
    display_research_page() is responsable for rendering the
    research with the report page.
    """

    st.title('Research Paper')
    st.info(Descriptions.RESEARCH_ABOUT)
    st.markdown('---')

    c1, c2, c3 = st.columns(3)

    c2.markdown("Please contact the contributors of the project in case you are interested in the research and full version of the software.")
    c2.write("          Via: [Lucas's Email](l.perasolovicentim@gmail.com)")

    # # Opening file from file path
    # with open('data/paper/MDP.pdf', "rb") as f:
    #     base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # # Embediding PDF in HTML
    # pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="700" type="application/pdf">'

    # # Displaying File
    # c2.markdown(pdf_display, unsafe_allow_html=True)
  