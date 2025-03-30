# Hunger Striker: A Food Security Prediction Tool

Hunger Striker is designed to assist policymakers, food suppliers, and researchers in addressing global food security challenges. By leveraging machine learning models trained on historical data, this tool provides valuable insights into two critical aspects of food security:

🔹 Note: The data used for both systems is synthetic, and predictions should not be interpreted as real-world forecasts.

**Read this first:**

Understanding the purpose of each file:

**1. Merged_data.csv and Food wsatage data.csv :** "Merged_data" is csv dataset for food insecurity model and "Food wastage data" is the dataset for foodwaste indicator model. 

 **2. Analysis_model_food_insecurity.ipynb and Analysis_model_foodwaste_indicator.ipynb :** These ipynb files show us the analysis including Data preprocessing, EDA, and comaprison of different model fits on the "Merged_data" and "Food wastage data" respectively.

**3. Prediction Data.csv, Waste classified data.csv and Waste prediction data.csv :**  "Prediction Data" is the final training data for the food insecurity model and, "Waste classified data" and "Waste prediction data" are the datasets including the  study variable and explanatory variables for the food waste indicator model.

**4. Pretraining_model_food_insecurity.ipynb and Pretraining_model_foodwaste_indicator.ipynb :** These ipynb files include the final training for the models which have been use for predicting "Food insecurity" and "Food wastage indicator" respectively.

**5. model.pkl and svm_model.pkl :** These include the pickle files format for the saved pretrained models used in predicting "Food insecurity" and "Food wastage indicator" respectively.

**6. wastage_model_features.pkl :** Just contains a list of feature names for the food wastage model in pickle format.

**7. app.py :** This is the python file used for deploying this website, loading and using the models and displaying the corresponding predictions according to user input

**8. streamlit-app-2025-03-30-19-03-46.webm :** Video of demo usage of website.

