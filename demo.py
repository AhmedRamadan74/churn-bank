import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
# streamlit layout

st.set_page_config(page_title="Churn bank",layout="wide")

st.title("Churn bank, By [Ahmed Ramadan](https://www.linkedin.com/in/ahmed-ramadan-18b873230/)")
st.header("About :")
st.markdown("Enter your bank's customers to know if customer will be leave the bank or he will be continuous to be a customer : ")
st.markdown("-----------------")

#input data
CreditScore=st.number_input("Enter Credit score of the customer :")
Geography=st.selectbox("Enter the country from which the customer belongs :",['France', 'Germany', 'Spain'])
Gender=st.selectbox("Enter gender of customer :",['Male', 'Female'])
Age=st.number_input("Enter age of the customer :")
Tenure=st.slider("Enter number of years for which the customer has been with the bank:",0,10)
Balance=st.number_input("Enter Bank balance of the customer :")
NumOfProducts=Tenure=st.slider("Enter number of bank products the customer is utilising:",0,4)
HasCrCard=st.selectbox("Enter the customer holds a credit card with the bank or not :",['Yes', 'No'])
IsActiveMember=st.selectbox("Enter the customer is an active member with the bank or not :",['Yes', 'No'])
EstimatedSalary=st.number_input("Enter estimated salary of the customer in Dollars :")
#make data frame
data={"CreditScore":CreditScore,
      "Geography":Geography,
      "Gender":Gender,
      "Age":Age,
      "Tenure":Tenure,
      "Balance":Balance,
      "NumOfProducts":NumOfProducts,
      "HasCrCard":HasCrCard,
      "IsActiveMember":IsActiveMember,
      "EstimatedSalary":EstimatedSalary}

df=pd.DataFrame(data,index=[0])

#load preprocessor and model

model = pickle.load(open(r"pipeline.pkl","rb"))

##preproccessing
df.HasCrCard=df.HasCrCard.map({"Yes":True,"No" :False})
df.IsActiveMember=df.IsActiveMember.map({"Yes":True,"No" :False})

#function predict
def prediction(df):
    value_predict=model.predict(df)[0]
    if value_predict ==1:
        return "This customer is not loyal and will want out of the service (Exited)"
    else:
        return "This customer is loyal and will stay in the service (Not Exited)"
    
#show the result
btn=st.button("Predict")

if btn:
    st.write(prediction(df))
