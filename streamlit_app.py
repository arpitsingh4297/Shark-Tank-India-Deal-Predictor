# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --------------------- Page Config ---------------------
st.set_page_config(
    page_title="Shark Tank India Deal Predictor",
    page_icon="shark",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------- Load Model ---------------------
@st.cache_resource
def load_model():
    model = joblib.load("sharktank_deal_predictor_final.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, expected_features = load_model()

# --------------------- Title & Intro ---------------------
st.title("Shark Tank India Deal Predictor")
st.markdown("### Will your startup get a deal from Aman, Anupam, Namita & team?")
st.image("https://upload.wikimedia.org/wikipedia/en/4/4b/Shark_Tank_India_logo.png", width=300)

# --------------------- Sidebar Inputs ---------------------
with st.sidebar:
    st.header("Pitch Details")
    
    ask_amount = st.number_input("Ask Amount (₹ Lakhs)", 10, 1000, 100)
    equity = st.slider("Equity Offered (%)", 1.0, 50.0, 5.0, 0.5)
    valuation_cr = st.number_input("Valuation Requested (₹ Crores)", 1.0, 200.0, 10.0)
    valuation = valuation_cr * 100  # convert to lakhs
    
    revenue = st.number_input("Yearly Revenue (₹ Lakhs)", 0, 20000, 200)
    monthly_sales = st.number_input("Monthly Sales (₹ Lakhs)", 0, 5000, 20)
    margin = st.slider("Gross Margin (%)", 0, 100, 45)
    
    col1, col2 = st.columns(2)
    with col1:
        patents = st.selectbox("Patents?", ["No", "Yes"])
    with col2:
        bootstrapped = st.selectbox("Bootstrapped?", ["Yes", "No"])
        
    presenters = st.slider("Number of Presenters", 1, 6, 2)
    age = st.selectbox("Pitchers Age Group", ["Young", "Middle", "Old"])
    sharks = st.slider("Sharks on Panel", 3, 7, 5)
    skus = st.number_input("Number of SKUs/Products", 0, 5000, 10)
    year = st.number_input("Started In (Year)", 1990, 2025, 2020)
    industry = st.selectbox("Industry", [
        "Food and Beverage", "Fashion", "Health/Wellness", "Beauty", "Technology",
        "Education", "Agriculture", "Electronics", "Home Decor", "Other"
    ])

    predict_btn = st.button("Predict Deal Chance", type="primary", use_container_width=True)

# --------------------- Prediction ---------------------
if predict_btn:
    # Feature engineering (must match training)
    data = {
        'Original Ask Amount': ask_amount,
        'Original Offered Equity': equity,
        'Valuation Requested': valuation,
        'Yearly Revenue': revenue,
        'Monthly Sales': monthly_sales,
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

    df_in = pd.DataFrame([data])
    df_in = pd.get_dummies(df_in, columns=['Industry'])

    # Align with training columns
    for col in expected_features:
        if col not in df_in.columns:
            df_in[col] = 0
    df_in = df_in[expected_features].astype(float)

    # Prediction
    probability = model.predict_proba(df_in)[0][1]
    prediction = model.predict(df_in)[0]

    # Results
    st.markdown(f"## {'DEAL LIKELY!' if prediction == 1 else 'NO DEAL'}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Deal Probability", f"{probability:.1%}")
    with col2:
        if probability >= 0.80:
            st.success("Extremely High Chance!")
        elif probability >= 0.60:
            st.info("Strong Pitch – Good Shot!")
        elif probability >= 0.40:
            st.warning("Possible – Improve Valuation")
        else:
            st.error("Low Chance – Reconsider Ask")

    # SHAP Explanation (Lightweight)
    with st.spinner("Explaining decision..."):
        explainer = shap.TreeExplainer(model.named_steps['clf'])
        shap_vals = explainer.shap_values(df_in)[0]

        # Top 10 factors
        shap_df = pd.DataFrame({
            "Feature": df_in.columns,
            "Impact": shap_vals
        }).sort_values(by="Impact", key=abs, ascending=False).head(10)

        st.subheader("Why this prediction?")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['green' if x > 0 else 'red' for x in shap_df['Impact']]
        ax.barh(shap_df['Feature'], shap_df['Impact'], color=colors, alpha=0.8)
        ax.set_xlabel("SHAP Value (Impact on Deal Chance)")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

    st.success("Prediction Complete!")

# Footer
st.markdown("---")
st.caption("Model: Tuned XGBoost | F1: 0.87 | AUC: 0.93 | Trained on 600+ real Shark Tank India pitches")