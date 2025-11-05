"""
🌸 Iris Species Predictor (Professional UI/UX Version)
Enhanced with responsive layout, modern visuals, and smooth interactions
"""

import numpy as np
import joblib
import requests
import streamlit as st
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================
# 🌸 Iris Predictor Logic
# ============================================================

class IrisPredictor:
    def __init__(self):
        try:
            self.model = joblib.load("iris_model.pkl")
            self.label_encoder = joblib.load("label_encoder.pkl")
        except Exception as e:
            st.error(f"⚠️ Error loading model files: {e}")
            raise e
        
        self.api_key = self._get_env_var("GEMINI_API_KEY")
        self.model_name = self._get_env_var("GEMINI_MODEL_NAME", "gemini-2.5-flash")

    def _get_env_var(self, var_name: str, default: Optional[str] = None) -> str:
        value = os.getenv(var_name, default)
        if value is None:
            st.warning(f"⚙️ Environment variable {var_name} not set.")
            return ""
        return value

    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        try:
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = self.model.predict(input_data)
            predicted_label = self.label_encoder.inverse_transform(prediction)[0]
            return predicted_label
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

    def get_species_info(self, species_name):
        if not self.api_key:
            return "⚠️ GEMINI_API_KEY not configured."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        prompt = (
            f"Provide a beautiful markdown-formatted explanation about the iris species '{species_name}'. "
            "Include: **overview**, **habitat**, **unique features**, and **fun facts**. "
            "Make it concise and visually appealing."
        )
        data = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"⚠️ API error {response.status_code}: {response.text}"
        except Exception as e:
            return f"⚠️ Connection error: {e}"

    def get_dataset_info(self):
        return {
            "total_samples": 150,
            "features": 4,
            "species_count": 3,
            "species_list": ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
        }

    def get_model_info(self):
        return {
            "algorithm": "Decision Tree Classifier",
            "training_data": "150 iris samples",
            "features": "4 numerical features",
            "target": "3 species classes",
            "accuracy": "High accuracy (~97%)",
            "validation": "Cross-validation used"
        }


# ============================================================
# 🎨 UI Manager
# ============================================================

class UIManager:

    @staticmethod
    def load_custom_css():
        st.markdown("""
        <style>
            /* Base Layout */
            .main-header {
                text-align: center;
                font-size: 2.2rem;
                color: #ff4b4b;
                font-weight: 700;
                margin-bottom: 20px;
            }
            .stButton>button {
                border-radius: 10px;
                background: linear-gradient(90deg, #ff4b4b, #ff7f50);
                color: white;
                padding: 0.6rem 1rem;
                font-weight: bold;
                border: none;
                transition: 0.3s;
            }
            .stButton>button:hover {
                transform: scale(1.03);
                background: linear-gradient(90deg, #ff6b6b, #ff9671);
            }
            .prediction-box {
                background-color: #ffffff10;
                padding: 1rem;
                border-radius: 15px;
                border: 1px solid #444;
                margin-top: 1rem;
                box-shadow: 0px 0px 15px rgba(255,75,75,0.2);
            }
            .info-box {
                background-color: #f9f9f9;
                padding: 1.2rem;
                border-radius: 12px;
                margin-top: 1rem;
            }
            .metric-container {
                display: flex;
                justify-content: space-around;
                margin-top: 10px;
            }
            .footer {
                text-align: center;
                color: #888;
                font-size: 0.8rem;
                margin-top: 3rem;
            }
        </style>
        """, unsafe_allow_html=True)

    # ----------------------------
    @staticmethod
    def render_sidebar(predictor):
        with st.sidebar:
            st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_versicolor_3.jpg", use_container_width=True)
            st.markdown("### 🌼 Iris Species Predictor")
            st.markdown("---")

            page = st.radio("📂 Navigation", ["🔮 Prediction", "ℹ️ About", "🤖 Model Info"], label_visibility="collapsed")
            st.markdown("---")

            st.markdown("### ✏️ Input Flower Measurements")
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
            petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

            st.markdown("#### 📏 Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sepal L", f"{sepal_length:.1f}cm")
                st.metric("Petal L", f"{petal_length:.1f}cm")
            with col2:
                st.metric("Sepal W", f"{sepal_width:.1f}cm")
                st.metric("Petal W", f"{petal_width:.1f}cm")

            return page, sepal_length, sepal_width, petal_length, petal_width

    # ----------------------------
    @staticmethod
    def render_prediction_page(predictor, sepal_length, sepal_width, petal_length, petal_width):
        st.markdown('<h1 class="main-header">🌸 Predict Your Iris Species</h1>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 🌼 Input Summary")
            st.markdown(f"""
            **Sepal Length:** {sepal_length} cm  
            **Sepal Width:** {sepal_width} cm  
            **Petal Length:** {petal_length} cm  
            **Petal Width:** {petal_width} cm
            """)

            if st.button("✨ Predict", use_container_width=True):
                with st.spinner("Analyzing your flower... 🌸"):
                    predicted_label = predictor.predict_species(sepal_length, sepal_width, petal_length, petal_width)

                if predicted_label:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.success(f"🎯 Predicted Species: **{predicted_label}**")
                    st.markdown('</div>', unsafe_allow_html=True)

                    with st.spinner("Fetching species info... 📚"):
                        info = predictor.get_species_info(predicted_label)
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown(info)
                    st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### 📈 Dataset Overview")
            info = predictor.get_dataset_info()
            st.metric("Samples", info["total_samples"])
            st.metric("Features", info["features"])
            st.metric("Species", info["species_count"])
            st.markdown("### 🌸 Types:")
            for s in info["species_list"]:
                st.markdown(f"- {s}")

    # ----------------------------
    @staticmethod
    def render_about_page():
        st.markdown('<h1 class="main-header">ℹ️ About This App</h1>', unsafe_allow_html=True)
        st.markdown("""
        This app predicts the **species of Iris flowers** using machine learning.  
        It’s built with **Streamlit**, **scikit-learn**, and the **Gemini API** for dynamic species descriptions.  

        #### 🧠 Dataset
        The classic **Iris dataset** contains 150 flower samples from 3 species:
        - Iris Setosa  
        - Iris Versicolor  
        - Iris Virginica  
        """)

    # ----------------------------
    @staticmethod
    def render_model_info_page(predictor):
        st.markdown('<h1 class="main-header">🤖 Model Information</h1>', unsafe_allow_html=True)
        info = predictor.get_model_info()

        st.markdown(f"""
        ### ⚙️ Model Details
        - Algorithm: {info['algorithm']}
        - Training Data: {info['training_data']}
        - Features: {info['features']}
        - Target: {info['target']}

        ### 📊 Performance
        - Accuracy: {info['accuracy']}
        - Validation: {info['validation']}
        """)

        st.progress(97)
        st.caption("Model Accuracy")

    # ----------------------------
    @staticmethod
    def render_footer():
        st.markdown('<div class="footer">Built with ❤️ using Streamlit | © 2025 Iris Predictor</div>', unsafe_allow_html=True)


# ============================================================
# 🚀 Main App
# ============================================================

def main():
    st.set_page_config(page_title="Iris Predictor", page_icon="🌸", layout="wide")
    predictor = IrisPredictor()
    UIManager.load_custom_css()

    page, sl, sw, pl, pw = UIManager.render_sidebar(predictor)

    if "Prediction" in page:
        UIManager.render_prediction_page(predictor, sl, sw, pl, pw)
    elif "About" in page:
        UIManager.render_about_page()
    elif "Model" in page:
        UIManager.render_model_info_page(predictor)

    UIManager.render_footer()


if __name__ == "__main__":
    main()
