import streamlit as st
import pandas as pd
import joblib

# Load the trained models
model_filename = 'model.pkl'
model = joblib.load(model_filename)

wastage_model = joblib.load('svm_model.pkl')

# Extract expected feature names from the models
expected_features = model.feature_names_in_
wastage_features = joblib.load('wastage_model_features.pkl') 

# Load the dataset
df = pd.read_csv('Prediction Data')
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

year_mapping = {2020: 2023, 2021: 2024, 2022: 2025}

# Apply the mapping to the "Year" column
df["Year"] = df["Year"].replace(year_mapping)

page = st.sidebar.selectbox("Choose Predictor", ["Food Shortage Prediction", "Food Wastage Level Prediction"])

if page == "Food Shortage Prediction":
    # Streamlit UI
    st.title('Food Shortage Prediction')
    st.write('Enter a country and year to predict the food shortage level.')
    
    # User inputs
    country = st.selectbox('Select Country', df['Country'].unique())
    year = st.selectbox('Select Year', sorted(df['Year'].unique()))
    
    # Predict button
    if st.button('Predict'):
        # Filter data based on user input
        input_data = df[(df['Country'] == country) & (df['Year'] == year)]
        
        if input_data.empty:
            st.write('No data available for the selected country and year.')
        else:
            # Drop unnecessary columns before prediction
            input_data = input_data.drop(columns=['Country', 'Year'], errors='ignore')
            
            # Ensure columns match model expectations
            input_data = input_data.reindex(columns=expected_features, fill_value=0)
            

elif page == "Food Wastage Level Prediction":
    st.header('Food Wastage Level Prediction')
    st.write('Enter a country and year to predict the food wastage level.')
    
    country = st.selectbox('Select Country', df['Country'].unique(), key='wastage_country')
    year = st.selectbox('Select Year', sorted(df['Year'].unique()), key='wastage_year')
    
    if st.button('Predict', key='wastage_predict'):
        input_data = df[(df['Country'] == country) & (df['Year'] == year)]
        
        if input_data.empty:
            st.write('No data available for the selected country and year.')
        else:
            input_data = input_data.drop(columns=['Country', 'Year'], errors='ignore')
            input_data = input_data.reindex(columns=wastage_features, fill_value=0)
            prediction = wastage_model.predict(input_data)
            st.write(f'Predicted Food Wastage Level: {prediction[0]}')

            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Display prediction
            st.write(f'Predicted Food Shortage chance: {round(prediction[0], 2) * 100}%')

