import streamlit as st

from medicare_app.predictor import FeatureValues, load_model, predict_risk

@st.cache_resource
def get_model():
    return load_model()


st.set_page_config(page_title="MediCare Diagnostics | Risk Screening", page_icon="❤️")
st.title("Heart Disease Risk Screening")
st.warning(
    "Educational screening only—not a medical diagnosis. Do not use this tool "
    "to make healthcare decisions or enter real patient information."
)
st.write(
    "Enter sample health attributes below. The model was trained on the UCI "
    "Cleveland heart disease dataset."
)

model = get_model()

with st.form("prediction_form"):
    st.subheader("Patient attributes (use sample data only)")
    left, right = st.columns(2)

    with left:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
        sex = st.selectbox(
            "Sex",
            options=[(0, "Female"), (1, "Male")],
            format_func=lambda item: item[1],
        )[0]
        cp = st.selectbox(
            "Chest pain type",
            options=[
                (1, "Typical angina"),
                (2, "Atypical angina"),
                (3, "Non-anginal pain"),
                (4, "Asymptomatic"),
            ],
            format_func=lambda item: item[1],
        )[0]
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)", min_value=60, max_value=250, value=120
        )
        chol = st.number_input("Cholesterol (mg/dL)", min_value=80, max_value=700, value=200)
        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dL",
            options=[(0, "No"), (1, "Yes")],
            format_func=lambda item: item[1],
        )[0]
        restecg = st.selectbox(
            "Resting ECG result",
            options=[
                (0, "Normal"),
                (1, "ST-T wave abnormality"),
                (2, "Left ventricular hypertrophy"),
            ],
            format_func=lambda item: item[1],
        )[0]

    with right:
        thalach = st.number_input(
            "Maximum heart rate achieved (bpm)", min_value=50, max_value=250, value=150
        )
        exang = st.selectbox(
            "Exercise-induced angina",
            options=[(0, "No"), (1, "Yes")],
            format_func=lambda item: item[1],
        )[0]
        oldpeak = st.number_input(
            "ST depression induced by exercise",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )
        slope = st.selectbox(
            "Peak exercise ST segment slope",
            options=[(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")],
            format_func=lambda item: item[1],
        )[0]
        ca_choice = st.selectbox(
            "Major vessels colored by fluoroscopy",
            options=[("Unknown", None), ("0", 0), ("1", 1), ("2", 2), ("3", 3)],
            format_func=lambda item: item[0],
        )
        thal_choice = st.selectbox(
            "Thalassemia result",
            options=[
                ("Unknown", None),
                ("Normal", 3),
                ("Fixed defect", 6),
                ("Reversible defect", 7),
            ],
            format_func=lambda item: item[0],
        )

    submitted = st.form_submit_button("Estimate risk")

if submitted:
    features: FeatureValues = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca_choice[1],
        "thal": thal_choice[1],
    }
    result = predict_risk(model, features)

    if result.flagged:
        st.error("The model flagged this sample as elevated risk.")
    else:
        st.success("The model did not flag this sample as elevated risk.")

    if result.positive_probability is not None:
        st.metric(
            "Model-estimated positive-class probability",
            f"{result.positive_probability:.1%}",
        )

    st.caption(
        "This result is a machine-learning estimate from a small educational dataset; "
        "it is not a diagnosis or a substitute for professional medical advice."
    )
