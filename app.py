import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pickle
from sklearn.preprocessing import OrdinalEncoder
encoder=OrdinalEncoder()
        
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report


data=pd.read_csv("C:\\Users\\User1\\Desktop\\BI\\ML\\ML Project1_BMP\\streamlit-prediction-app\\bank-marketing.csv")
def main():
    st.title("      Bank Marketing Prediction     ")
    st.spinner()


    htk=  """
    <div style="background-color:crimson;padding:0px">
    <h2 style="color:white;text-align:center;">Customer Response Prediction App </h2>
    </div>
    """
    st.markdown(htk,unsafe_allow_html=True)


    model=st.sidebar.selectbox(
    "Select an ML Model",
    ("Logistic Regression", "Random Forest")
    )
    if model=="Logistic Regression":
        # st.header("Top 3 feature columns are taken here ,after performing EDA and Feature Engineering.")
        klm= """
    
    <h3 style="color:black;text-align:center;"> Top 3 feature columns are taken here ,after performing EDA and Feature Engineering. </h3>
    
    """
        st.markdown(klm,unsafe_allow_html=True)
        housing=st.sidebar.radio("Do you have Housing Loan?", ('yes' , 'no'))
        loan=st.sidebar.radio("Do you have Personal Loan?",('yes', 'no'))
        contact=st.sidebar.radio("How do you prefer to communicate",("unknown",'Telephone', 'Cellular'))
       
        
        dict={"yes":'1',"no":'0'}
        for  i, j in dict.items():
            housing=housing.replace(i,j)
            loan=loan.replace(i,j)

        contact_dict={"unknown":'2',"Cellular":'0',"Telephone":'1'}
        for  i, j in contact_dict.items():
            contact=contact.replace(i,j)
        
        
        with open("lr.pkl",'rb') as f:
            lr=pickle.load(f)
        
        res=lr.predict([[int(housing),int(loan),int(contact)]])
        
        
      
       
        
    else:
        age=st.slider("Enter age of the customer",18,95)
        age=int(age)
        age=int(scaler.fit_transform([[age]]))
        job=st.selectbox("Enter the type of job customer do",('management' ,'technician', 'entrepreneur', 'blue-collar' ,'unknown',
        'retired', 'admin.' ,'services', 'self-employed', 'unemployed' ,'housemaid','student'))
        job=int(encoder.fit_transform([[job]]))
        salary=st.text_input("Enter the salary of the customer",0,120000)
        
        salary=int(salary)
        if salary>120000 or salary<0:
            st.text("Please enter a number between  0 & 120000 ")
        salary=int(scaler.fit_transform([[salary]]))
        marital=st.selectbox("what is customer's marital status?",('married','single' ,'divorced'))
        marital=int(encoder.fit_transform([[marital]]))
        education=st.selectbox("Enter customer's education level",('tertiary', 'secondary' ,'unknown' ,'primary'))
        education=int(encoder.fit_transform([[education]]))
        targeted=st.selectbox("Do the customer have target?", ('yes' , 'no'))
        targeted=int(encoder.fit_transform([[targeted]]))
        default=st.selectbox("Do the customer have credit in default?", ('yes' , 'no'))
        default=int(encoder.fit_transform([[default]]))
        balance=st.slider("Enter the customers balance",min_value=-8019,max_value=102127)
        balance=int(balance)
        if balance>102127 or balance<-8019:
             st.text("Please enter a number between -8019 & 102127")
        balance=int(scaler.fit_transform([[balance]]))
        housing=st.selectbox("Do the customer have Housing Loan?", ('yes' , 'no'))
        housing=int(encoder.fit_transform([[housing]]))
        loan=st.selectbox("Do the customer have Personal Loan?",('yes', 'no'))
        loan=int(encoder.fit_transform([[loan]]))
        contact=st.radio("How do you prefer to communicate",("unknown",'Telephone', 'Cellular'))
        contact=int(encoder.fit_transform([[contact]]))
        day=st.slider("Enter Day",1,31)
        day=int(day)
        day=int(scaler.fit_transform([[day]]))
        month=st.selectbox("Which month the customer was last contacted in?",('jan',"Feb","March",'April',"May","June","July","Aug","Sep","oct","Nov","Dec"))
        month=int(encoder.fit_transform([[month]]))
        duration=st.text_input("Enter last contact duration with the customer in sec?",0,4918)
        
        duration=int(duration)
        if duration>4918 or duration<0:
             st.text("Please enter an number between 0 & 4918")
        duration=int(scaler.fit_transform([[duration]]))
        campaign=st.slider("Enter number of contacts performed during this campaign and for this client",1,63)
        campaign=int(campaign)
        
        campaign=int(scaler.fit_transform([[campaign]]))

        with open("rf.pkl",'rb') as f:
            rf=pickle.load(f)
        res=rf.predict([[age,job,salary,marital,education,targeted,default,balance,housing,loan,contact,day,month,duration,campaign]])
        
        

        
    return res

    

if __name__=='__main__':
    result=main()
    Res=str(int(result))
    dict={"yes":'1',"no":'0'}    
    for i,j in dict.items():
        Res=Res.replace(j,i)
    
            
    
    st.subheader("The predicted response of customer or client to subscribe a term deposit is")
    st.success(Res)
    
    if st.checkbox("About Data "):
        st.text("Please visit the link provided here to know the complete details of the dataset;")
        st.text(" __Data set link__: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing ")
    
    
    if st.button("Thanks") :
        st.text("Thank you for visiting  and happy learning :)")
        st.balloons()
   