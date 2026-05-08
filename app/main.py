import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode and modern look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #0068c9;
        color: white;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135692.png", width=100)
    st.markdown("## Navigation")
    
    selected = option_menu(
        menu_title=None,
        options=[
            "Home", 
            "Dataset Explorer", 
            "Live NLP Pipeline",
            "Model Classification",
            "Explainable AI",
            "Resume Ranking",
            "Model Comparison"
        ],
        icons=[
            "house", 
            "database", 
            "funnel",
            "robot",
            "search",
            "bar-chart-line",
            "graph-up"
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#0068c9"},
        }
    )

    st.markdown("---")
    st.markdown("Developed for NLP University Project")

# Routing
if selected == "Home":
    import app.pages.page_home as Home
    Home.show()
elif selected == "Dataset Explorer":
    import app.pages.page_dataset as Dataset_Explorer
    Dataset_Explorer.show()
elif selected == "Live NLP Pipeline":
    import app.pages.page_pipeline as NLP_Pipeline
    NLP_Pipeline.show()
elif selected == "Model Classification":
    import app.pages.page_classification as Classification
    Classification.show()
elif selected == "Explainable AI":
    import app.pages.page_explainability as Explainability
    Explainability.show()
elif selected == "Resume Ranking":
    import app.pages.page_ranking as Resume_Ranking
    Resume_Ranking.show()
elif selected == "Model Comparison":
    import app.pages.page_comparison as Model_Comparison
    Model_Comparison.show()
