import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the trained models
model_filename = 'model.pkl'
model = joblib.load(model_filename)

wastage_model = joblib.load('svm_model.pkl')

# Extract expected feature names from the models
expected_features = model.feature_names_in_
wastage_features = joblib.load('wastage_model_features.pkl') 

# Load the dataset-1
df = pd.read_csv('Prediction Data')
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

year_mapping = {2020: 2023, 2021: 2024, 2022: 2025}

# Apply the mapping to the "Year" column
df["Year"] = df["Year"].replace(year_mapping)

# Load dataset-2
df1=pd.read_csv("Waste prediction data")
df2=pd.read_csv("Waste classified data")

y_wastage = pd.read_csv('Waste classified data')
y_wastage = y_wastage.iloc[1::2].values.ravel()

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
            prediction = model.predict(input_data)
            st.write(f'Predicted Food Shortage chance: {round(prediction[0], 2) * 100}%')

elif page == "Food Wastage Level Prediction":
    st.header('Food Wastage Level Prediction')
    st.write('Enter a country and year to predict the food wastage level.')

    country = st.selectbox('Select Country', df1['country'].unique(), key='wastage_country')
    year = st.selectbox('Select Year', sorted(df1['Year'].unique()), key='wastage_year')

    if st.button('Predict', key='wastage_predict'):
        input_data = df1[(df1['country'] == country) & (df1['Year'] == year)]
        
        if input_data.empty:
            st.write('No data available for the selected country and year.')
        else:
            input_data = input_data.drop(columns=['country', 'Year'], errors='ignore')

            # Apply Standard Scaling
            scaler = StandardScaler()
            full_scaled = scaler.fit_transform(df1.drop(columns=['country', 'Year'], errors='ignore'))
            input_scaled = scaler.transform(input_data)

            # Apply LDA transformation
            lda = LinearDiscriminantAnalysis(n_components=2)
            X_lda = lda.fit_transform(full_scaled, y_wastage)  # Train once
            input_lda = lda.transform(input_scaled)  # Transform new input

            # Prediction
            prediction = wastage_model.predict(input_lda)
            st.write(f'Predicted Food Wastage Level: {prediction[0]}')
            
