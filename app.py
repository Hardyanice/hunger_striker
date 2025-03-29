import streamlit as st
import pandas as pd
import numpy as np
import joblib  # If model is saved as a pickle file
import sqlite3

# Load the trained ML model
model = joblib.load("model.pkl")  # Change this to the correct model path

# Database connection
conn = sqlite3.connect("your_database.db")  # Change to your actual database

# Streamlit UI
st.title("Food Shortage Early Warning System")

st.sidebar.header("User Input")
country = st.sidebar.text_input("Enter Country Name")
year = st.sidebar.number_input("Enter Year", min_value=2000, max_value=2100, step=1)

if st.sidebar.button("Predict"):
    # Retrieve corresponding row from the database
    query = f"SELECT * FROM your_table WHERE country = '{country}' AND year = {year}"
    df = pd.read_sql(query, conn)

    if not df.empty:
        # Preprocess the data if needed
        X = df.drop(columns=["country", "year"])  # Adjust based on dataset structure

        # Make predictions
        predictions = model.predict(X)

        # Display results
        st.subheader("Predicted Food Shortage Probability")
        st.write(predictions)

        # Optional: Visualize predictions
        st.bar_chart(predictions)

    else:
        st.error("No data found for the given country and year.")

st.sidebar.write("Note: Predictions are based on historical data and trends.")

# Close database connection
conn.close()

