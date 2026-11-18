from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Breast Cancer Classification", page_icon="🔬")
st.title("Breast Cancer Classification")
st.write("Test a trained machine learning model using an example row or your own values.")
st.warning("Educational demonstration only. This prediction is not a medical diagnosis.")

folder = Path(__file__).parent

if not (folder / "best_model.pkl").exists():
    st.error("Missing best_model.pkl. Add your saved model to this GitHub repository.")
    st.stop()

try:
    model = joblib.load(folder / "best_model.pkl")
except Exception as error:
    st.error("Could not load the model. Check that the scikit-learn version matches training.")
    st.exception(error)
    st.stop()

samples = pd.read_csv(folder / "sample_rows.csv")
# These are the names used in sample_rows.csv and on the screen.
features = ["f" + str(i) for i in range(1, 31)]

# Use the exact names and order learned by the saved model.
model_features = list(getattr(model, "feature_names_in_", []))
if not model_features and hasattr(model, "steps"):
    model_features = list(getattr(model.steps[0][1], "feature_names_in_", []))
if not model_features:
    st.error("This model does not provide its training feature names. Please check the training notebook.")
    st.stop()
if len(model_features) != len(features):
    st.error("The saved model expects a different number of features.")
    st.stop()

st.subheader("Choose input")
mode = st.radio("Input method", ["Ready-made example", "Enter values manually"])

if mode == "Ready-made example":
    choice = st.selectbox("Select an example", samples["example"].tolist())
    selected = samples[samples["example"] == choice].iloc[0]
    values = selected[features].astype(float)
    st.write("Input features (30 columns):")
    st.dataframe(pd.DataFrame([values]), hide_index=True)
    st.caption("Actual label in the original dataset: " +
               ("Malignant (0)" if selected["actual_target"] == 0 else "Benign (1)"))
else:
    st.write("Change any of the 30 feature values below. The example values are starting points only.")
    start = samples.iloc[0]
    values = {}
    with st.form("input_form"):
        cols = st.columns(3)
        for i, name in enumerate(features):
            with cols[i % 3]:
                values[name] = st.number_input(name, value=float(start[name]), format="%.6f")
        submitted = st.form_submit_button("Predict")

if mode == "Ready-made example":
    submitted = st.button("Predict")

if submitted:
    # Preserve feature values in dataset order, but use names from training.
    input_data = pd.DataFrame([[values[name] for name in features]], columns=model_features)
    try:
        result = int(model.predict(input_data)[0])
        if result == 0:
            st.error("Model prediction: Malignant (0)")
        elif result == 1:
            st.success("Model prediction: Benign (1)")
        else:
            st.error("Unexpected model output. Check target labels used during training.")
        st.caption("This model predicts from numerical features; it cannot confirm a clinical diagnosis.")
    except Exception as error:
        st.error("Prediction failed. Check that this model was trained on the same 30 columns, in the same order.")
        st.exception(error)
