
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.impute import SimpleImputer

# Load trained model pipeline
try:
    with open("rf_pipeline.pkl", "rb") as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error("Pipeline file (rf_pipeline.pkl) not found. Please train and save the pipeline first (e.g., by running the relevant cells in the notebook).")
    st.stop()

st.set_page_config(page_title="MTCT Risk Predictor", page_icon="🍼", layout="centered")

st.title("🍼 Mother-to-Child Transmission (MTCT) Risk Predictor")
st.markdown("Predict the probability of HIV transmission to infant based on maternal factors.")

# Display class order for debugging
st.sidebar.subheader("Model Class Order")
st.sidebar.write(pipe.classes_)

# --- Input fields ---
viral_load = st.number_input("Maternal Viral Load (copies/ml)", min_value=0, value=1000, step=100)
art_duration = st.number_input("ART Duration (months)", min_value=0, value=12, step=1)
active_pmtct = st.selectbox("Active in PMTCT program?", ["Yes", "No", "Pregnant", "Breastfeeding", "nan"])
tpt_outcome = st.selectbox("Prophylaxis/TPT Outcome", ["Treatment completed", "Discontinued", "Not Started", "nan"])
# Removed 'months of prescription' as it was not used in the model features

# --- Feature vector ---
input_dict = {
    "last vl": viral_load,
    "art_duration_months": art_duration,
    "active in pmtct": active_pmtct,
    "tpt outcome": tpt_outcome,
    # Removed 'months of prescription'
}

# Create DataFrame from input
X_new = pd.DataFrame([input_dict])

# Use the pipeline to preprocess and predict
if st.button("Predict MTCT Risk"):
    # The pipeline handles all preprocessing (imputation, one-hot encoding)
    # Note: The pipeline expects columns in the order they were in X_train_pipe
    # Need to make sure the input DataFrame has the expected columns before passing to pipeline
    # The columns expected by the pipeline's preprocessor are the original ones:
    # 'last vl', 'art_duration_months', 'vl_suppressed', 'active in pmtct', 'tpt outcome'
    # We need to add 'vl_suppressed' and ensure column order if the pipeline expects it
    # However, the pipeline approach with ColumnTransformer handles column order based on definition,
    # so just ensuring the original columns are present is key.

    # Add vl_suppressed feature (this needs to be done before passing to the pipeline)
    X_new["vl_suppressed"] = X_new["last vl"].apply(
        lambda x: 1 if pd.notnull(x) and x < 1000 else (0 if pd.notnull(x) else np.nan)
    )


    # Ensure input DataFrame has the columns expected by the pipeline
    # Based on the pipeline definition in gt-mndD2PNnm, the preprocessor expects:
    # numerical_pipe = ["last vl", "art_duration_months", "vl_suppressed"]
    # categorical_pipe = ["active in pmtct", "tpt outcome"]
    # We need to make sure X_new has these columns before passing to pipe.predict_proba

    # Reorder and select only the columns the pipeline expects
    # This list must match the columns used to fit the pipeline's preprocessor
    expected_cols_for_pipeline = ["last vl", "art_duration_months", "vl_suppressed", "active in pmtct", "tpt outcome"]
    X_new_processed_for_pipe = X_new[expected_cols_for_pipeline]


    # Use the pipeline to predict probabilities
    probabilities = pipe.predict_proba(X_new_processed_for_pipe)[0]

    # Find the index for the 'High Risk' class
    try:
        high_risk_index = list(pipe.classes_).index("High Risk")
    except ValueError:
        st.error("'High Risk' class not found in model classes.")
        st.stop()

    prob = probabilities[high_risk_index] # Probability of High Risk

    # --- Custom threshold ---
    threshold = 0.5  # set your own cutoff here

    label = "High Risk" if prob >= threshold else "Low Risk"
    # If the model's predict method gives the correct label, you can use that instead of the threshold logic
    # model_predicted_label = pipe.predict(X_new_processed_for_pipe)[0]
    # st.subheader(f"Model's Direct Prediction: **{model_predicted_label}**")


    st.subheader(f"Predicted Risk: **{label}**")
    st.metric("Probability of High Risk", f"{prob*100:.2f}%")

    # Simple interpretation
    st.markdown("### 🔍 Interpretation")
    if viral_load > 1000:
        st.write("- High maternal viral load contributes to increased MTCT risk.")
    if art_duration < 6:
        st.write("- Short ART duration increases risk.")
    if active_pmtct == "No":
        st.write("- Not being in PMTCT program increases risk.")
    if tpt_outcome == "Not Started":
        st.write("- Lack of TPT is a risk factor.")

    st.success("✅ Use this information for clinical decision support, not as a diagnostic tool.")
