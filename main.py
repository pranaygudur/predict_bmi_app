import streamlit as st
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import  LabelEncoder
#import xgboost as xgb
#import numpy as np
st.header("BMI Quote Prediction App")
st.text_input("Enter your Name(Optional): ", key="name")
data = pd.read_csv("Dummy-Data.csv")
#load label encoder
#encoder = LabelEncoder()
#encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
#best_xgboost_model = xgb.XGBRegressor()
#best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    data

st.subheader("Please provide your inputs below to generate quote instantly")
left_column, right_column = st.columns(2)
with left_column:
    input_gender = st.radio(
        'Gender:',
       ['Male','Female'])


input_age= st.slider('Age', 0, 100, 1)
input_height = st.text_input("""Height(e.g for 5'9" inches enter as 509)""",500)
input_weight = st.slider('Weight(in lbs)', 0, 800, 1)

def category(age,wt,ht,gender):
    pound_to_kg = 0.45
    inches_to_meters = 0.0254

    bmi = wt*pound_to_kg/(((int(str(ht)[0])*12 + int(str(ht)[-2:]))*inches_to_meters)**2)

    if gender=='Male':
        if (age>=18 and age<=39) and (bmi<17.49 or bmi>38.5):
            return 750
        elif (age>=40 and age<=59) and (bmi<18.49 or bmi>38.5):
            return 1000
        elif (age>=60) and (bmi<18.49 or bmi>38.5): 
            return 2000
        else:
            return 500
    else:
        return category(age,bmi,'Male')*(1-0.1)

reason_code_dictionary = {750:"Age is between 18 to 39 and 'BMI' is either less than 17.49 or greater than 38.5",
               1000:"Age is between 40 to 59 and 'BMI' is either less than 18.49 or greater then 38.5”",
               2000:"Age is greater than 60 and 'BMI' is either less than 18.49 or greater than 38.5",
               500:"BMI is in right range"}        
    
if st.button('Generate your quote today!'):
    ct_quote = category(input_age,input_weight,input_height,'Male')
    to_print = reason_code_dictionary[ct_quote]

    if input_gender == 'Female':
        #print(to_print)
        ct_quote = ct_quote*0.9
    st.write(f"The quote is $ {ct_quote}")
    st.write(f"Reason code :{to_print}")

