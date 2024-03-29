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


data=pd.read_csv("bank-marketing.csv")
def main():
    
    

    st.title("      Bank Marketing Prediction     ")
    


    htk=  """
    <div style="background-color:crimson;padding:0px">
    <h3 style="color:white;text-align:center;">Customer Response Prediction App </h3>
    </div>
    """
    st.markdown(htk,unsafe_allow_html=True)
    
    if st.button("Problem Statement"):
       
        st.markdown(""" The  goal is to make a predictive model to predict if the customer will respond positively to the
            campaign organised by a portugese bank institution. Often, more than one contact to the same client was required, in order to access 
        if the product (bank term deposit) would be subscribed or not subscribed.
        This is a tedious task to do and will consume much time ,the model can help in avoiding both.
         """)
    if st.checkbox("Data Description"):
        st.text(
        """ The data is related with direct marketing campaigns of a Portuguese banking institution.
        The marketing campaigns were based on phone calls.All the information can be found in the link given below.
        """)
        st.text("PLEASE REFER TO THE DATA SET LINK BEFORE ACTUALLY GOING FORWARD WITH THE APP TO GET A BETTER UNDERSTANDING AND CLEAR IDEA.")
        st.text("Note:'bank-full.csv' data inside 'bank.zip' is used.")
        
        link='[Data set link](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)'
        st.markdown(link,unsafe_allow_html=True)

    model=st.sidebar.selectbox(
    "Select an ML Model",
    ("select","Logistic Regression", "Random Forest")
    )
    if model=="Logistic Regression":
        if st.checkbox("Modeling"):
        
            klm= """
    
    <h4 style="color:red;text-align:center;"> Top 3 feature columns are taken here to prophesy final output ,after performing EDA and Feature Engineering. </h4>
    
    """
            st.markdown(klm,unsafe_allow_html=True)
        
        housing=st.sidebar.selectbox("Does the customer have Housing Loan?", ("select",'yes' , 'no'))
        loan=st.sidebar.selectbox("Does the customer have Personal Loan?",("select",'yes', 'no'))
        contact=st.sidebar.selectbox("How does the customer prefer to communicate",("select","unknown",'Telephone', 'Cellular'))
        if housing!="select" and  loan!="select" and contact!="select":
        
            dict={"yes":'1',"no":'0'}
            for  i, j in dict.items():
                housing=housing.replace(i,j)
                loan=loan.replace(i,j)

            contact_dict={"unknown":'2',"Cellular":'0',"Telephone":'1'}
            for  i, j in contact_dict.items():
                contact=contact.replace(i,j)
        
        
            with open("lr.pkl",'rb') as f:
                lr=pickle.load(f)
        
            res=str(lr.predict([[int(housing),int(loan),int(contact)]]))
            dict={"yes":'1',"no":'0'}    
            for i,j in dict.items():
                res=res.replace(j,i)
        else:
            res="None"
    

          
      
       
        
    elif model=="Random Forest":
        age=st.slider("Enter age of the customer",18,95)
        age=int(age)
        age=int(scaler.fit_transform([[age]]))
        job=st.selectbox("Enter the type of job customer do",("select",'management' ,'technician', 'entrepreneur', 'blue-collar' ,'unknown',
        'retired', 'admin.' ,'services', 'self-employed', 'unemployed' ,'housemaid','student'))
        job=int(encoder.fit_transform([[job]]))
        salary=st.number_input("Enter the salary of the customer")
        
        
       
        if  salary>120000 or salary<0:
            st.warning("Invalid, Please enter an amount between  0 & 120000 ")
            
           
        salary=int(scaler.fit_transform([[salary]]))
        marital=st.selectbox("what is customer's marital status?",("select",'married','single' ,'divorced'))
        
        education=st.selectbox("Enter customer's education level",("select",'tertiary', 'secondary' ,'unknown' ,'primary'))
        
        targeted=st.selectbox("Does the customer have target?", ("select",'yes' , 'no'))
        
        default=st.selectbox("Does the customer have credit in default?", ("select",'yes' , 'no'))
        
        balance=st.text_input("Enter the customers balance",0)
        if  not balance :
            st.warning("Please enter balance")
            
        else:
            if balance.isalpha() and balance.isalnum():
                st.warning("Please enter an integer number")
                pass
                
        
        
            else:
                balance=int(balance)
                if balance>102127 or balance<-8019:
                    st.warning("Please enter a balance amount between -8019 & 102127")
            
        
                balance=int(scaler.fit_transform([[balance]]))
        housing=st.selectbox("Do the customer have Housing Loan?", ("select",'yes' , 'no'))
        
        loan=st.selectbox("Do the customer have Personal Loan?",("select",'yes', 'no'))
        
        contact=st.radio("How do you prefer to communicate",("select","unknown",'Telephone', 'Cellular'))
        month=st.selectbox("Which month the customer was last contacted in?",("select",'Jan',"Feb","March",'April',"May","June","July","Aug","Sep","Oct","Nov","Dec"))
        
        if month  in ["Jan","March","May","July","Aug","Oct","Dec"] :
            day=st.slider("Enter Day the customer was contacted ",1,31)
            day=int(day)
            
        elif month=="Feb":
            day=st.slider("Enter Day the customer was contacted ",1,29)
            day=int(day)
        else:
            day=st.slider("Enter the Day the customer was contacted ",1,30)
            day=int(day)

        
        day=int(scaler.fit_transform([[day]]))
        duration=st.text_input("Enter last contact duration with the customer in sec?",0,4918)
        if not duration:
            st.warning("Enter Duration Period")
            
        else:
            if duration.isalpha() and duration.isalnum():
                st.warning("Please enter an integer number")
                pass
            else:
                duration=int(duration)
                if duration>4918 or duration<0:
                    st.warning("Please enter an number between 0 & 4918")
                duration=int(scaler.fit_transform([[duration]]))
        campaign=st.slider("Enter number of contacts performed during this campaign and for this client",1,63)
        campaign=int(campaign)
        
        campaign=int(scaler.fit_transform([[campaign]]))
        if marital!="select" and education!="select" and targeted!="select" and default!="select" and housing!="select" and loan!="select" and contact!="select" and month!="select":
            marital=int(encoder.fit_transform([[marital]]))
            education=int(encoder.fit_transform([[education]]))
            targeted=int(encoder.fit_transform([[targeted]]))
            default=int(encoder.fit_transform([[default]]))
            housing=int(encoder.fit_transform([[housing]]))
            loan=int(encoder.fit_transform([[loan]]))
            contact=int(encoder.fit_transform([[contact]]))
            month=int(encoder.fit_transform([[month]]))

            with open("rf.pkl",'rb') as f:
                rf=pickle.load(f)
            res=rf.predict([[age,job,salary,marital,education,targeted,default,balance,housing,loan,contact,day,month,duration,campaign]])
            res=str(res)
            dict={"yes":'1',"no":'0'}    
            for i,j in dict.items():
                res=res.replace(j,i)
        else:
            res="None"

        

    else:
        res="None"
           
    return res

    

if __name__=='__main__':
    Res=main()
    # Res=str(int(result))
    # dict={"yes":'1',"no":'0'}    
    # for i,j in dict.items():
    #     Res=Res.replace(j,i)
    
            
    if st.sidebar.button("Show Prediction"):
        st.sidebar.subheader("The predicted response of customer or client to subscribe a term deposit is")
        st.sidebar.success(Res)
    
    
    
    
    if st.button("Thanks") :
        st.text("Thank you for visiting  and happy learning :)")
        st.balloons()
    
   