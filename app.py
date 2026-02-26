import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: white;
}
h1 {
    text-align: center;
    font-size: 40px;
}
div.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #ff1e1e;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("model.sav")
    columns = joblib.load("columns.pkl")
    return model, columns

# ---------------- Main App ----------------
def main():

    st.title("🏦 Loan Approval Prediction")
    st.markdown("---")

    model, cols = load_model()

    with st.form("loan_form"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            credit_history = st.selectbox("Credit History", ["Yes", "No"])

        with col2:
            property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
            income = st.number_input("Applicant Income", min_value=0.0)
            loan_amount = st.number_input("Loan Amount", min_value=0.0)
            loan_term = st.number_input("Loan Amount Term", min_value=0.0)
            dependents = st.number_input("Dependents", min_value=0.0)

        submit = st.form_submit_button("Predict")

    if submit:

        # -------- Encoding --------
        input_data = {
            "ApplicantIncome": income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Dependents": dependents,
            "Gender": 1 if gender == "Male" else 0,
            "Married": 1 if married == "Yes" else 0,
            "Education": 1 if education == "Graduate" else 0,
            "Self_Employed": 1 if self_employed == "Yes" else 0,
            "Credit_History": 1 if credit_history == "Yes" else 0,
            "Property_Area": 0 if property_area == "Rural"
                              else 1 if property_area == "Semiurban"
                              else 2
        }

        # -------- Create DataFrame Safely --------
        features = pd.DataFrame([input_data])

        # Ensure same column order as training
        features = features.reindex(columns=cols, fill_value=0)

        # -------- Prediction --------
        prediction = model.predict(features)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("💚 You Are Eligible for Loan")
        else:
            st.error("❌ You Are Not Eligible for Loan")

    st.markdown(
        "<div style='text-align:center; padding:20px;'>Developed with 💖 by <b>Navya Gupta</b></div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
