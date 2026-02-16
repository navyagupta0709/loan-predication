import streamlit as st
import numpy as np
import joblib
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# ---------------- Load Model ----------------
def load_model():
    return joblib.load("model.sav")

# ---------------- Main App ----------------
def main():

    st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

    st.title("🏦 Loan Approval Prediction")

    st.markdown("---")

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
            income = st.number_input("Applicant Income", min_value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0)
            loan_term = st.number_input("Loan Amount Term", min_value=0)
            dependents = st.number_input("Dependents", min_value=0)

        submit = st.form_submit_button("Predict")

    # ---------------- Prediction ----------------
    if submit:

        gender_val = 1 if gender == "Male" else 0
        married_val = 1 if married == "Yes" else 0
        education_val = 1 if education == "Graduate" else 0
        employed_val = 1 if self_employed == "Yes" else 0
        credit_val = 1 if credit_history == "Yes" else 0

        property_val = 0 if property_area == "Rural" else 1 if property_area == "Semiurban" else 2

        features = np.array([[income,
                              loan_amount,
                              loan_term,
                              dependents,
                              gender_val,
                              married_val,
                              education_val,
                              employed_val,
                              credit_val,
                              property_val]])

        model = load_model()
        prediction = model.predict(features)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("💚 You Are Eligible for Loan")
        else:
            st.error("❌ You Are Not Eligible for Loan")

    # ---------------- Footer ----------------
    st.markdown(
        "<div style='text-align:center; padding:20px;'>Developed with 💖 by <b>Navya Gupta</b></div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
