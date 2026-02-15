import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODEL ----------------
def load_model():
    model = joblib.load("model.sav")
    return model

# ---------------- MAIN FUNCTION ----------------
def main():

    st.title("🏦 Loan Approval Prediction System")

    st.write("Fill the details below to check loan approval status.")

    # ----------- INPUT FIELDS -----------

    gender = st.radio("Gender", ["Male", "Female"])
    married = st.radio("Married", ["Yes", "No"])
    education = st.radio("Education", ["Graduate", "Not Graduate"])
    employed = st.radio("Self Employed", ["Yes", "No"])
    property_area = st.radio("Property Area", ["Rural", "Semiurban", "Urban"])

    applicant_income = st.number_input("Applicant Income", min_value=0.0)
    coapplicant_income = st.number_input("Co-Applicant Income", min_value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
    loan_term = st.number_input("Loan Amount Term", min_value=0.0)
    credit_history = st.slider("Credit History", 0, 1)
    dependents = st.slider("Dependents", 0, 3)

    # ----------- MANUAL ENCODING (SAFE) -----------

    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    employed = 1 if employed == "Yes" else 0

    property_dict = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    property_area = property_dict[property_area]

    # ----------- PREDICTION -----------
if st.button("Submit"):

    model = load_model()

    # Manual encoding
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    education_val = 1 if education == "Graduate" else 0
    employed_val = 1 if employed == "Yes" else 0
    property_dict = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    property_val = property_dict[property_area]

    # IMPORTANT: Same order as training
    features = np.array([[applicant_income,
                          coapplicant_income,
                          loan_amount,
                          loan_term,
                          gender_val,
                          married_val,
                          dependents,
                          education_val,
                          employed_val,
                          credit_history,
                          property_val]])

    prediction = model.predict(features)

    st.subheader("Result:")

    if prediction[0] == 1:
        st.success("🎉 Your Loan will get Approved.")
    else:
        st.error("❌ Your Loan will NOT get Approved.")
   

# ---------------- FOOTER ----------------
st.markdown(
    "<div style='text-align:center; padding:20px;'>Developed with ❤️ by <b>Navya Gupta</b></div>",
    unsafe_allow_html=True
)

# ---------------- RUN APP ----------------
if __name_ == "__main__":
    main()
