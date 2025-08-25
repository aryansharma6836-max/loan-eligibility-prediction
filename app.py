import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("🏦 Loan Eligibility Prediction")
st.write("Enter your details below to check loan eligibility:")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("dataset.csv")
    return data

data = load_data()

# Train model
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Input fields
income = st.number_input("Applicant Income", min_value=0)
credit_score = st.slider("Credit Score", 300, 900, 650)
employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
loan_amount = st.number_input("Loan Amount", min_value=0)

# Encode employment status
emp_map = {"Employed": 0, "Self-Employed": 1, "Unemployed": 2}
employment_encoded = emp_map[employment_status]

# Predict
if st.button("Check Eligibility"):
    features = [[income, credit_score, employment_encoded, loan_amount]]
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("✅ Eligible for Loan")
    else:
        st.error("❌ Not Eligible for Loan")

# Show model accuracy
st.write(f"📊 Model Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")
