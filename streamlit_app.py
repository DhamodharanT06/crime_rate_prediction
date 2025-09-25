
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load('crime_prediction_model.pkl')
features = joblib.load('model_features.pkl')  

df = pd.read_csv('crime_dataset_india.csv')

le_desc = LabelEncoder().fit(df['Crime Description'].astype(str))
le_dom = LabelEncoder().fit(df['Crime Domain'].astype(str))

def preprocess_data(X):
    X_processed = pd.DataFrame(X, columns=features)
    for col in ['Crime Description', 'Crime Domain']:
        le = le_desc if col == 'Crime Description' else le_dom
        X_processed[col] = le.transform(X_processed[col].astype(str))
    return X_processed.astype(float)

st.title("🚨 Crime Safety Prediction")
st.write("Enter crime-related details to check if the area is **safe** or **unsafe**.")

input_data = {}
for feature in features:
    if feature == 'City_Encoded':
        cities = df['City'].unique()
        city = st.selectbox('City', cities)
        input_data[feature] = int(pd.Series(cities).tolist().index(city))

    elif feature == 'Crime Description':
        desc = st.selectbox('Crime Description', le_desc.classes_)
        input_data[feature] = desc

    elif feature == 'Crime Domain':
        dom = st.selectbox('Crime Domain', le_dom.classes_)
        input_data[feature] = dom

    else:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("🔍 Predict"):
    input_df = pd.DataFrame([input_data])
    processed_input = preprocess_data(input_df)
    prediction = model.predict(processed_input)[0]
    st.success("✅ Prediction: SAFE" if prediction == 1 else "❌ Prediction: UNSAFE")
