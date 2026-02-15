
import streamlit as st 
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def load_model(modelfile):
    loaded_model = joblib.load(modelfile)
    return loaded_model


def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Loan Approval Prediction 💵🏦 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.sidebar.write(""" ## How does it work ❓
The loan approval prediction model uses your credit score, income, education, loan amount and other various factor  to determine whether your loan will be approved or not.  It outputs a "yes" or "no" decision.
                """)
    
    with st.sidebar.expander(" ℹ️ Information", expanded=True):
        st.write("""
**Applicant income**: This refers to the amount of money that the person applying for the loan earns on a regular basis.

**Co-applicant income**: If the loan applicant has a co-applicant (such as a spouse or partner), their income will also be taken into consideration by the lender.

**Loan amount**: This is the total amount of money that the loan applicant is requesting to borrow.

**Loan amount term**: This refers to the length of time over which the loan will be repaid.

**Gender**: This is the loan applicant's gender, which is recorded for statistical purposes only. 

**Married**: This refers to the loan applicant's marital status. Lenders may use this information to evaluate the applicant's financial stability and ability to repay the loan.

**Dependents**: This refers to the number of people that the loan applicant supports financially (such as children or elderly parents).

**Education**: This refers to the loan applicant's level of education.

**Employed**: This refers to the loan applicant's employment status.

**Credit history**: This refers to the loan applicant's past credit behavior, including any loans or credit cards they may have had in the past.
        """)


    with st.expander(" ℹ️ About", expanded=True):
        st.write("""
                Loan Prediction is very helpful for employee of banks 
                as well as for the applicant also.
                Dream housing Finance Company 
                deals in all loans. They have presence across all urban, 
                semi urban and rural areas. Customer first apply for loan 
                after that company or bank validates the customer 
                eligibility for loan. Company or bank wants to automate 
                the loan eligibility process (real time) based on customer 
                details provided while filling application form. These 
                details are Gender, Marital Status, Education, Number of 
                Dependents, Income, Loan Amount, Credit History and 
                other
        """)



    with st.form("my_form"):
        col4,col6 = st.columns([2,2])
        with col4:
            education = st.radio("Education", ['Graduate','Not Graduate'])
            married = st.radio("Married", ['Yes','No'])
            loanAmountTerm = st.number_input("Loan Amount Term", 10.0,600.0)
            applicantIncome = st.number_input("Applicant Income", 150.0,100000.0)
            propertyArea = st.radio("Property Area", ['Ruler','Semiurban','Urban'])
        with col6:
            employed = st.radio("Self Employed", ['Yes','No'])
            gender = st.radio("Gender", ['Male','Female'])
            coApplicantIncome = st.number_input("Co Applicant Income", 0.0,50000.0)
            loanAmount = st.number_input("Loan Amount", 10.0,1000.0)
            creditHistory = str(st.slider("Credit History",1,3000))
            dependents = st.slider("Dependents", 0,4)

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            num_data = [applicantIncome, coApplicantIncome, loanAmount, loanAmountTerm]
            cat_data = [gender, married, dependents, education, employed,creditHistory, propertyArea]
            test=[]
            column=['applicagoinIncome','coApplicantIncome','loanAmount','loanAmountTrem','gender','married','dependent','education','employed','creditHistory','propertyArea']
            test = le.fit_transform(cat_data).tolist()
            feature_list = num_data + test
            # df = pd.DataFrame([feature_list],columns=column)
            single_pred = np.array(feature_list).reshape(1,-1)
            # single_pred = np.array(feature_list)
            # st.write(feature_list) 
            loaded_model = load_model('model.sav')
            prediction = loaded_model.predict(single_pred)
            st.write('''
		    ## Results 🔍 
		    ''')
            if prediction[0] == "0":
                st.write("Your Loan will get Approved.")
            else: 
                st.write("Your Loan will not get Approved.")



hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)



if __name__ == '__main__':
	main()

footer="""<style>
a:link , a:visited{
color: white;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}



if __name__ == '__main__':
	main()

ooter = """
<style>
.footer {
    position: relative;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0e1117;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>

<div class="footer">
    Developed with 💖 by <b>Navya Gupta</b>
</div>
"""

st.markdown(footer, unsafe_allow_html=True
