"""
streamlit_app.py — Simple web UI for the Titanic Survival Prediction model.

Run with:
    streamlit run streamlit_app.py
"""
import streamlit as st
from predict import predict_survival, VALID_TITLES

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("Titanic Survival Predictor")
st.write(
    "Enter a passenger's details below to predict whether they would have survived "
    "the Titanic disaster, using a tuned Random Forest classifier."
)

with st.form("passenger_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Ticket Class (Pclass)", options=[1, 2, 3], index=2,
                               help="1 = 1st class, 2 = 2nd class, 3 = 3rd class")
        sex = st.selectbox("Sex", options=["male", "female"])
        age = st.slider("Age", min_value=0, max_value=100, value=30)
        title = st.selectbox("Title", options=VALID_TITLES,
                              help="Extracted from passenger name (e.g. Mr, Mrs, Miss, Master)")

    with col2:
        sibsp = st.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
        cabin_known = st.checkbox("Cabin number known?", value=False)
        ticket_freq = st.number_input("People sharing this ticket", min_value=1, max_value=10, value=1)

    submitted = st.form_submit_button("Predict Survival")

if submitted:
    passenger = {
        "Pclass": pclass, "Sex": sex, "Age": float(age), "SibSp": int(sibsp),
        "Parch": int(parch), "Fare": float(fare), "Title": title,
        "CabinKnown": cabin_known, "TicketFreq": int(ticket_freq)
    }

    try:
        result = predict_survival(passenger)
        st.divider()

        if result["prediction"] == 1:
            st.success(f"✅ Prediction: **{result['prediction_label']}**")
        else:
            st.error(f"❌ Prediction: **{result['prediction_label']}**")

        st.metric("Survival Probability", f"{result['survival_probability'] * 100:.1f}%")
        st.progress(result["survival_probability"])

    except ValueError as e:
        st.error(f"Invalid input: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption(
    "Model: Random Forest Classifier, tuned via GridSearchCV/RandomizedSearchCV with 5-fold CV. "
    "Test accuracy: 79.3% | ROC-AUC: 0.850. Built during the AnalystLab Africa internship."
)