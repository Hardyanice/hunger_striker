import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_filename = 'model.pkl'
model = joblib.load(model_filename)

# Load the dataset
df = pd.read_csv('Prediction Data')
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

# Ensure feature alignment
expected_features = joblib.load('model_features.pkl')  # Load expected feature names

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
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display prediction
        st.write(f'Predicted Food Shortage Level: {prediction[0]}')
