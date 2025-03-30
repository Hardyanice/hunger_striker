import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the trained models
shortage_model = joblib.load('model.pkl')
wastage_model = joblib.load('svm_model.pkl')

# Extract expected feature names from the shortage model
shortage_features = shortage_model.feature_names_in_

# Load the dataset
df = pd.read_csv('Prediction Data')
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

# Load the scaler
scaler = StandardScaler()

# Load the classified labels for LDA
y_wastage = pd.read_csv('Waste classified data')
y_wastage = y_wastage.iloc[1::2].values.ravel()

# Apply Standard Scaling to the entire dataset
X_scaled = scaler.fit_transform(df.drop(columns=['Country', 'Year'], errors='ignore'))

"""
# Apply LDA transformation
lda = LinearDiscriminantAnalysis(n_components=2)
if X_scaled.shape[0] == y_wastage.shape[0]:
    X_lda = lda.fit_transform(X_scaled, y_wastage)
else:
    raise ValueError(f"Mismatch: X_scaled has {X_scaled.shape[0]} rows, but y_wastage has {y_wastage.shape[0]} rows")


# Mapping for food wastage levels
wastage_mapping = {0: 'Very Low', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}

"""


# Streamlit UI with enhanced professional styling
st.set_page_config(page_title='Food Security Prediction', layout='wide')
st.title('🌍 Food Security Prediction Dashboard')
st.markdown("---")

# Sidebar navigation with improved styling
st.sidebar.header("Navigation")
st.sidebar.markdown("Select a predictor from below:")
page = st.sidebar.radio("Choose Page", ["Home","📉 Food Shortage Prediction", "🍽️ Food Wastage Level Prediction"])

if page == "🏠 Home":
    st.header("🏠 Welcome to the Food Security Prediction Dashboard!")
    st.write("""
        This tool provides insights into **food shortages** and **food wastage levels** across different countries.
        
        🌱 **Food Shortage Prediction**: Predicts the probability of food shortages in a given country and year.  
        🍽️ **Food Wastage Level Prediction**: Classifies food wastage levels from Very Low to Critical.  

        Use the **sidebar** to select a prediction tool.""")  
        
elif page == "📉 Food Shortage Prediction":
    st.header('📉 Food Shortage Prediction')
    st.write('Select a country and year to predict the food shortage level.')
    
    country = st.selectbox('🌍 Select Country', df['Country'].unique())
    year = st.selectbox('📅 Select Year', sorted(df['Year'].unique()))
    
    if st.button('🔍 Predict'):
        input_data = df[(df['Country'] == country) & (df['Year'] == year)]
        
        if input_data.empty:
            st.warning('⚠️ No data available for the selected country and year.')
        else:
            input_data = input_data.drop(columns=['Country', 'Year'], errors='ignore')
            input_data = input_data.reindex(columns=shortage_features, fill_value=0)
            prediction = shortage_model.predict(input_data)
            st.success(f'📊 Predicted Food Shortage Chance: **{round(prediction[0], 2) * 100}%**')

"""
elif page == "🍽️ Food Wastage Level Prediction":
    st.header('🍽️ Food Wastage Level Prediction')
    st.write('Select a country and year to predict the food wastage level.')
    
    country = st.selectbox('🌍 Select Country', df['Country'].unique(), key='wastage_country')
    year = st.selectbox('📅 Select Year', sorted(df['Year'].unique()), key='wastage_year')
    
    if st.button('🔍 Predict', key='wastage_predict'):
        input_data = df[(df['Country'] == country) & (df['Year'] == year)]
        
        if input_data.empty:
            st.warning('⚠️ No data available for the selected country and year.')
        else:
            input_data = input_data.drop(columns=['Country', 'Year'], errors='ignore')
            input_scaled = scaler.transform(input_data)
            input_lda = lda.transform(input_scaled)
            prediction = wastage_model.predict(input_lda)
            mapped_prediction = wastage_mapping.get(prediction[0], 'Unknown')
            st.success(f'🍽️ Predicted Food Wastage Level: **{mapped_prediction}**')

"""
