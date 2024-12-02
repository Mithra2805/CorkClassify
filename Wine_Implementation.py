import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set Streamlit title and header
st.title("Wine Quality Prediction using Random Forest")
st.header("Predicting wine quality and classifying as Good or Bad")

# Load the dataset with proper handling for separator and column names
url = r"C:\Users\admin\Documents\Projects\Wine Quality Prediction Project\WineQT.csv"

# Try loading the dataset with proper separator
try:
    # Ensure that we specify the correct separator (semicolon or comma based on the actual dataset format)
    df = pd.read_csv(url, sep=';')  # Use semicolon as separator (if using comma, change to sep=',')
    st.write("Dataset loaded successfully!")
except Exception as e:
    st.write(f"Error loading dataset: {e}")
    st.stop()  # Stop further execution if dataset loading fails

# Display the first few rows and column names of the dataset for inspection
st.write("First few rows of the dataset:")
st.write(df.head())

# Display column names for debugging
st.write(f"Original Columns in the dataset: {df.columns.tolist()}")

# If columns are incorrectly loaded as a single column, split them manually
if len(df.columns) == 1 and ',' in df.columns[0]:
    # Split the first column by comma to fix column names
    df = df[df.columns[0].str.split(',', expand=True)]  # Split the first column into multiple columns
    df.columns = df.iloc[0]  # Set the first row as column names
    df = df.drop(0)  # Drop the first row since it's now the header
    st.write("Column names fixed by splitting values by comma.")

# Clean up column names (strip leading/trailing spaces)
df.columns = df.columns.str.strip()

# Display cleaned columns
st.write(f"Cleaned Columns in the dataset: {df.columns.tolist()}")

# Check if 'quality' column exists exactly as expected
if 'quality' not in df.columns:
    st.write("Error: 'quality' column is missing from the dataset!")
    st.write(f"Available columns: {df.columns.tolist()}")
    st.stop()  # Stop execution if 'quality' column is missing
else:
    st.write("'quality' column found!")

# Remove 'Id' column if it exists (it's not useful for prediction)
if 'Id' in df.columns:
    df.drop(columns='Id', inplace=True)
    st.write("'Id' column removed.")

# Display the corrected columns
st.write(f"Columns after cleaning: {df.columns.tolist()}")

# Splitting the dataset into features (X) and target (y)
X = df.drop(columns='quality')  # Features (all columns except 'quality')
y = df['quality']  # Target (wine quality)

# Standardize the features (important for models like Random Forest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display evaluation results
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.write("Classification Report:")
st.text(classification_rep)

# Feature Importance visualization
st.write("Feature Importance:")
importance = rf_classifier.feature_importances_
features = X.columns

# Create a DataFrame of features and their importance
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
st.pyplot()

# Allow the user to input new wine data for prediction
st.header("Predict Wine Quality")

# Create input fields for the user to enter values for each feature
def user_input():
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.slider(
            feature, 
            min_value=int(df[feature].min()), 
            max_value=int(df[feature].max()), 
            value=int(df[feature].mean())
        )
    return pd.DataFrame(input_data, index=[0])

# Get user input
user_data = user_input()

# Standardize the user input
user_data_scaled = scaler.transform(user_data)

# Predict wine quality
user_prediction = rf_classifier.predict(user_data_scaled)

# Convert the prediction into "Good" or "Bad" classification
wine_quality = user_prediction[0]
if wine_quality >= 6:
    wine_status = "Good"
else:
    wine_status = "Bad"

# Display the prediction
st.write(f"The predicted wine quality is: {wine_quality}")
st.write(f"Therefore, the wine is classified as: **{wine_status}**")




import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set Streamlit title and header
st.title("Wine Quality Prediction using Random Forest")
st.header("Predicting wine quality and classifying as Good or Bad")

# Load the dataset with proper handling for separator and column names
url = r"C:\Users\admin\Documents\Projects\Wine Quality Prediction Project\WineQT.csv"

# Try loading the dataset with proper separator
try:
    # Ensure that we specify the correct separator (semicolon or comma based on the actual dataset format)
    df = pd.read_csv(url, sep=';')  # Use semicolon as separator
    st.write("Dataset loaded successfully!")
except Exception as e:
    st.write(f"Error loading dataset: {e}")
    st.stop()  # Stop further execution if dataset loading fails

# Display the first few rows and column names of the dataset for inspection
st.write("First few rows of the dataset:")
st.write(df.head())

# Display column names for debugging
st.write(f"Original Columns in the dataset: {df.columns.tolist()}")

# Check if columns are being loaded correctly
if len(df.columns) == 1 and ',' in df.columns[0]:
    # If columns are merged into one, split by comma
    df = df[df.columns[0].str.split(',', expand=True)]  # Split columns by comma
    df.columns = df.iloc[0]  # Set the first row as column names
    df = df.drop(0)  # Drop the first row since it's now the header
    st.write("Column names fixed by splitting values by comma.")
else:
    # If columns are already properly loaded, just clean up the column names
    df.columns = df.columns.str.strip()  # Remove any leading or trailing spaces

# Display cleaned columns
st.write(f"Cleaned Columns in the dataset: {df.columns.tolist()}")

# Check if 'quality' column exists exactly as expected
if 'quality' not in df.columns:
    st.write("Error: 'quality' column is missing from the dataset!")
    st.write(f"Available columns: {df.columns.tolist()}")
    st.stop()  # Stop execution if 'quality' column is missing
else:
    st.write("'quality' column found!")

# Remove 'Id' column if it exists (it's not useful for prediction)
if 'Id' in df.columns:
    df.drop(columns='Id', inplace=True)
    st.write("'Id' column removed.")

# Display the corrected columns
st.write(f"Columns after cleaning: {df.columns.tolist()}")

# Splitting the dataset into features (X) and target (y)
X = df.drop(columns='quality')  # Features (all columns except 'quality')
y = df['quality']  # Target (wine quality)

# Standardize the features (important for models like Random Forest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display evaluation results
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.write("Classification Report:")
st.text(classification_rep)

# Feature Importance visualization
st.write("Feature Importance:")
importance = rf_classifier.feature_importances_
features = X.columns

# Create a DataFrame of features and their importance
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
st.pyplot()

# Allow the user to input new wine data for prediction
st.header("Predict Wine Quality")

# Create input fields for the user to enter values for each feature
def user_input():
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.slider(
            feature, 
            min_value=int(df[feature].min()), 
            max_value=int(df[feature].max()), 
            value=int(df[feature].mean())
        )
    return pd.DataFrame(input_data, index=[0])

# Get user input
user_data = user_input()

# Standardize the user input
user_data_scaled = scaler.transform(user_data)

# Predict wine quality
user_prediction = rf_classifier.predict(user_data_scaled)

# Convert the prediction into "Good" or "Bad" classification
wine_quality = user_prediction[0]
if wine_quality >= 6:
    wine_status = "Good"
else:
    wine_status = "Bad"

# Display the prediction
st.write(f"The predicted wine quality is: {wine_quality}")
st.write(f"Therefore, the wine is classified as: **{wine_status}**")
