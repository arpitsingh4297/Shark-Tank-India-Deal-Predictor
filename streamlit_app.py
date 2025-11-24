# streamlit_app.py - CLOUD-PROOF VERSION (No SHAP, Uses XGBoost Importance)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Shark Tank India Deal Predictor",
    page_icon="🦈",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    model = joblib.load("sharktank_deal_predictor_final.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, expected_features = load_model()

st.title("🦈 Shark Tank India Deal Predictor")
st.markdown("### Will your startup get a deal from Aman, Anupam, Namita & team?")

# Sidebar inputs
with st.sidebar:
    st.header("Pitch Details")
    ask = st.number_input("Ask Amount (₹ Lakhs)", 10, 1000, 100)
    equity = st.slider("Equity Offered (%)", 1.0, 50.0, 5.0)
    val_cr = st.number_input("Valuation Requested (₹ Crores)", 1.0, 500.0, 10.0)
    valuation = val_cr * 100  # to lakhs
    revenue = st.number_input("Yearly Revenue (₹ Lakhs)", 0, 50000, 150)
    margin = st.slider("Gross Margin (%)", 0, 100, 40)
    patents = st.selectbox("Has Patents?", ["No", "Yes"])
    bootstrapped = st.selectbox("Bootstrapped?", ["Yes", "No"])
    presenters = st.slider("Number of Presenters", 1, 6, 2)
    age = st.selectbox("Pitchers Age Group", ["Young", "Middle", "Old"])
    sharks = st.slider("Sharks on Panel", 3, 7, 5)
    skus = st.number_input("Number of SKUs/Products", 0, 1000, 10)
    year = st.number_input("Started In (Year)", 1990, 2025, 2020)
    industry = st.selectbox("Industry", [
        "Food and Beverage", "Fashion", "Health/Wellness", "Beauty", "Technology",
        "Education", "Agriculture", "Electronics", "Home Decor", "Other"
    ])
    predict_btn = st.button("Predict Deal Chance", type="primary", use_container_width=True)

if predict_btn:
    # Feature engineering (exact match to training)
    data = {
        'Original Ask Amount': ask,
        'Original Offered Equity': equity,
        'Valuation Requested': valuation,
        'Yearly Revenue': revenue,
        'Monthly Sales': revenue / 12,
        'Gross Margin': margin,
        'Has Patents': 1 if patents == "Yes" else 0,
        'Bootstrapped': 1 if bootstrapped == "Yes" else 0,
        'Number of Presenters': presenters,
        'Pitchers Average Age': {"Young": 0, "Middle": 1, "Old": 2}[age],
        'Total Sharks Present': sharks,
        'SKUs': skus,
        'Started in': year,
        'Ask Valuation Multiple': valuation / (revenue + 1),
        'Profitable': 1 if (margin > 30 and revenue > 50) else 0,
        'Industry': industry
    }

    df_input = pd.DataFrame([data])
    df_input = pd.get_dummies(df_input, columns=['Industry'])

    # Align with expected features
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[expected_features].astype(float)

    # Predict
    with st.spinner("Analyzing your pitch..."):
        prob = model.predict_proba(df_input)[0][1]
        pred = model.predict(df_input)[0]

    # Results
    st.markdown(f"## {'🟢 DEAL LIKELY!' if pred == 1 else '🔴 NO DEAL'}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Deal Probability", f"{prob:.1%}")
    with col2:
        if prob >= 0.8:
            st.success("🎉 Extremely High Chance!")
        elif prob >= 0.6:
            st.info("👍 Strong Pitch – Good Shot!")
        elif prob >= 0.4:
            st.warning("⚠️ Possible – Improve Numbers")
        else:
            st.error("❌ Low Chance – Revisit Ask/Valuation")

    # XGBoost Native Explanation (No SHAP – Bulletproof!)
    st.subheader("Why this prediction? (Top Factors)")
    xgb_model = model.named_steps['clf']
    importance_df = pd.DataFrame({
        'Feature': expected_features,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(8)
    
    st.bar_chart(importance_df.set_index('Feature'))
    
    st.caption("Higher bars = stronger influence on getting a deal. (Powered by XGBoost)")

st.markdown("---")
st.caption("Model: Tuned XGBoost | F1: 0.87 | AUC: 0.93 | Trained on 634+ Shark Tank India pitches (Seasons 1-4)")
