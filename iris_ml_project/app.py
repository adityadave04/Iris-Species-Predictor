"""
Main Streamlit application for Iris Species Predictor
This is the main entry point that orchestrates the UI and backend logic
"""

import streamlit as st
from backend import IrisPredictor, UIManager

# Page configuration
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize backend components
@st.cache_resource
def load_predictor():
    """Load the predictor model (cached for performance)"""
    return IrisPredictor()

# Load CSS styles
UIManager.load_custom_css()

# Initialize predictor
predictor = load_predictor()

# Render sidebar and get user inputs
page, sepal_length, sepal_width, petal_length, petal_width = UIManager.render_sidebar(predictor)

# Main content area based on selected page
if "Prediction" in page:
    UIManager.render_prediction_page(predictor, sepal_length, sepal_width, petal_length, petal_width)

elif "About" in page:
    UIManager.render_about_page()

elif "Model" in page:
    UIManager.render_model_info_page(predictor)
