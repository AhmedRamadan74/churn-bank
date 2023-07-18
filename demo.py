import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import sklearn

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

preprocessor=pickle.load(open(r"/home/ramy/سطح المكتب/ahmed/Data Scince and analysis and AI/Data science cousre Epsilon/Machine learning/projects/Churn project/preprocessor.pkl",'rb'))
model = pickle.load(open(r"/home/ramy/سطح المكتب/ahmed/Data Scince and analysis and AI/Data science cousre Epsilon/Machine learning/projects/Churn project/model.pkl","rb"))

##preproccessing
df.HasCrCard=df.HasCrCard.map({"Yes":True,"No" :False})
df.IsActiveMember=df.IsActiveMember.map({"Yes":True,"No" :False})

#transform
df_test=preprocessor.transform(df)

#predict
result=model.predict(df_test)

#show the result
btn=st.button("Predict")

if btn:
    if result==1:
        st.write(f"The customer will be leave")
    else:
        st.write(f"The customer will be continuous")