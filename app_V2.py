import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the predictor model from a pickle file
model = pickle.load(open('model_V2.pkl', 'rb'))

# Load the encoder dictionary from a pickle file
with open('encoder_V2.pkl', 'rb') as pkl_file:
    encoder_dict = pickle.load(pkl_file)

def encode_features(df, encoder_dict):
    # For each categorical feature, apply the encoding
    category_col = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
    for col in category_col:
        if col in encoder_dict: 
            le = LabelEncoder()
            le.classes_ = np.array(encoder_dict[col], dtype=object)  # Load the encoder classes for this column

            # Handle unknown categories by using 'transform' method and a lambda function
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            df[col] = le.transform(df[col])
    return df

def main():
    st.title("Income Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Income Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    age = st.text_input("Age","0")
    workclass = st.selectbox("Working Class", ["Federal-gov","Local-gov","Never-worked","Private","Self-emp-inc","Self-emp-not-inc","State-gov","Without-pay"])
    education = st.selectbox("Education",["10th","11th","12th","1st-4th","5th-6th","7th-8th","9th","Assoc-acdm","Assoc-voc","Bachelors","Doctorate","HS-grad","Masters","Preschool","Prof-school","Some-college"])
    marital_status = st.selectbox("Marital Status",["Divorced","Married-AF-spouse","Married-civ-spouse","Married-spouse-absent","Never-married","Separated","Widowed"])
    occupation = st.selectbox("Occupation",["Adm-clerical","Armed-Forces","Craft-repair","Exec-managerial","Farming-fishing","Handlers-cleaners","Machine-op-inspct","Other-service","Priv-house-serv","Prof-specialty","Protective-serv","Sales","Tech-support","Transport-moving"])
    relationship = st.selectbox("Relationship",["Husband","Not-in-family","Other-relative","Own-child","Unmarried","Wife"])
    race = st.selectbox("Race",["Amer Indian Eskimo","Asian Pac Islander","Black","Other","White"])
    gender = st.selectbox("Gender",["Female","Male"])
    capital_gain = st.text_input("Capital Gain","0")
    capital_loss = st.text_input("Capital Loss","0")
    hours_per_week = st.text_input("Hours per week","0")
    nativecountry = st.selectbox("Native Country",["Cambodia","Canada","China","Columbia","Cuba","Dominican Republic","Ecuador","El Salvadorr","England","France","Germany","Greece","Guatemala","Haiti","Netherlands","Honduras","HongKong","Hungary","India","Iran","Ireland","Italy","Jamaica","Japan","Laos","Mexico","Nicaragua","Outlying-US(Guam-USVI-etc)","Peru","Philippines","Poland","Portugal","Puerto-Rico","Scotland","South","Taiwan","Thailand","Trinadad&Tobago","United States","Vietnam","Yugoslavia"])

    if st.button("Predict"):

        data = {'age': int(age), 'workclass': workclass, 'education': education, 'maritalstatus': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 'capitalgain': int(capital_gain), 'capitalloss': int(capital_loss), 'hoursperweek': int(hours_per_week), 'nativecountry': nativecountry}
        # print(data)
        # Convert the data into a DataFrame for easier manipulation
        df = pd.DataFrame([data])

        # Encode the categorical columns
        df = encode_features(df, encoder_dict)

        # Now, all your features should be numerical, and you can attempt prediction
        features_list = df.values
        prediction = model.predict(features_list)

        output = int(prediction[0])
        if output == 1:
            text = ">50K"
        else:
            text = "<=50K"

        st.success('Employee Income is {}'.format(text))

if __name__=='__main__':
    main()
