import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="AI vs Human Detection", layout="wide")

st.title("🧠 AI vs Human Content Detection (2026 Edition)")
st.markdown("End-to-End ML Deployment with Multiple Models")

# ---------------------------------
# Load Models
# ---------------------------------
@st.cache_resource
def load_models():
    logistic = joblib.load("logistic_model.pkl")
    rf = joblib.load("random_forest_model.pkl")
    optimized = joblib.load("optimized_logistic_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return logistic, rf, optimized, le

log_model, rf_model, opt_model, le = load_models()

# ---------------------------------
# Load Dataset for Visualization
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ai_human_detection_v1.csv")

df = load_data()

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.header("⚙ Model Selection")

model_option = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest", "Optimized Logistic"]
)

if model_option == "Logistic Regression":
    model = log_model
elif model_option == "Random Forest":
    model = rf_model
else:
    model = opt_model

# ---------------------------------
# Text Prediction Section
# ---------------------------------
st.subheader("✍ Enter Text for Prediction")

user_input = st.text_area("Type or paste text here...")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])
        probabilities = model.predict_proba([user_input])

        label = le.inverse_transform(prediction)[0]
        confidence = np.max(probabilities)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2%}")

        # Probability Bar Chart
        prob_df = pd.DataFrame({
            "Class": le.classes_,
            "Probability": probabilities[0]
        })

        fig = px.bar(
            prob_df,
            x="Class",
            y="Probability",
            title="Prediction Probabilities",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Data Visualization Section
# ---------------------------------
st.subheader("📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        df,
        x="human_or_ai",
        color="human_or_ai",
        title="Class Distribution",
        text_auto=True
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.box(
        df,
        x="human_or_ai",
        y="word_count",
        color="human_or_ai",
        title="Word Count Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------
# Language Distribution
# ---------------------------------
st.subheader("🌍 Language Distribution")

fig3 = px.histogram(
    df,
    x="language",
    color="human_or_ai",
    barmode="group",
    title="Class Distribution by Language"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------
# Domain Distribution
# ---------------------------------
st.subheader("🏷 Domain Analysis")

fig4 = px.histogram(
    df,
    x="domain",
    color="human_or_ai",
    title="Distribution Across Domains"
)

fig4.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.markdown("Developed as an End-to-End ML Project 🚀")