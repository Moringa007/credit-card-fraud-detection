import pandas as pd
import joblib
import os
import warnings
import tkinter as tk
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Suppress all warnings
warnings.filterwarnings("ignore")

def preprocess_data_for_prediction(df):
    """
    Preprocesses the dataframe by handling date/time features,
    and creating new features.
    """

    # Drop unnecessary columns and convert date/time features
    drop_columns = ['Unnamed: 0', 'cc_num', 'first', 'last', 'unix_time', 'trans_num', 'street', 'city']
    df.drop(drop_columns, axis=1, inplace=True, errors='ignore')

    # Convert trans_date_trans_time to datetime and extract features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    # Extract hour, day of week, month, and age from date/time
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df['year'] = df['trans_date_trans_time'].dt.year

    # Drop original datetime column
    df.drop(columns=['trans_date_trans_time'], inplace=True)

    # Convert DOB and calculate age
    df['dob'] = pd.to_datetime(df['dob'])
    df['year_dob'] = df['dob'].dt.year
    df['age'] = df['year'] - df['year_dob']

    # Add age_group feature using bins
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # Fill missing value in 'age_group' with the mode
    # Ensure the mode is calculated the df 'age_group' column after creation
    df['age_group'] = df['age_group'].fillna(df['age_group'].mode()[0])

    #print(df.info())

    selected_feature_cols = ['amt', 'lat', 'long', 'city_pop', 'day', 'month', 'hour', 'age', 'merchant', 'category', 'job', 'age_group']

    # Ensure 'is_fraud' is not in the features for prediction
    if 'is_fraud' in df.columns:
        df.drop(columns=['is_fraud'], inplace=True)
    X = df[selected_feature_cols]

    print(f"Preprocessed DataFrame shape: {X.shape}\n")
    print(X.info())


    # Get categorical features for tree-based models
    # categorical_features = ['merchant', 'category', 'job', 'age_group']
    # numerical_features = ['amt', 'lat', 'long', 'city_pop', 'day', 'month', 'hour', 'age']

    # # Label Encoding
    # encoders = {col: LabelEncoder() for col in categorical_features}
    # for col in categorical_features:
    #     # Handle new categories not seen in training
    #     X[col] = X[col].astype(str)
    #     unknown_category = f"UNKNOWN_{col}"
    #     le_classes = list(encoders[col].classes_)
    #     if unknown_category not in le_classes:
    #         encoders[col].classes_ = np.append(encoders[col].classes_, unknown_category)
    #     X[col] = X[col].apply(lambda x: x if x in le_classes else unknown_category)
    #     X[col] = encoders[col].transform(X[col])
    # joblib.dump(encoders, 'checkpoints/encoders.joblib')

    # # Scaling
    # scaler = StandardScaler()
    # X[numerical_features] = scaler.transform(X[numerical_features])
    # print(X.info())
    

    return X


def run_prediction_demo():
    """
    Loads the best model, performs fraud detection, and exports results.
    """
    try:
        # Find the best model saved in the checkpoints folder
        best_model_path = None
        for file in os.listdir('checkpoints'):
            if file.endswith('_model.joblib'):
                best_model_path = os.path.join('checkpoints', file)
                print(f"Loading model from: {best_model_path}")
                break

        if not best_model_path:
            print("No trained model found. Please run 'python train.py' first.")
            return

        # Load the model
        model = joblib.load(best_model_path)
        
        # Load the test data
        test_df_original = pd.read_csv('data/fraudTest.csv')
        test_df = test_df_original.copy()
        test_df = preprocess_data_for_prediction(test_df)

        # Get categorical features for tree-based models
        categorical_features = ['merchant', 'category', 'job', 'age_group']
        test_df[categorical_features] = test_df[categorical_features].astype('category')
        
        # Handle scaling if the best model was RandomForest
        if 'randomforest' in best_model_path:
            scaler = joblib.load('checkpoints/scaler.joblib')
            X_test_scaled = scaler.transform(test_df)
            predictions = model.predict(X_test_scaled)
        else:
            predictions = model.predict(test_df)

        # Add predictions to the original dataframe
        test_df_original['prediction'] = predictions

        # Save the output to a new CSV file
        output_file = 'data/fraudTest_predictions.csv'
        test_df_original.to_csv(output_file, index=False)
        print(f"Fraud detection complete. Predictions saved to '{output_file}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_prediction_demo()