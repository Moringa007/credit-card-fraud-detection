import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from tqdm import tqdm
import joblib
import os
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

def preprocess_data(df):
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
    X = df[selected_feature_cols]
    y = df['is_fraud']

    # Get categorical features for tree-based models
    categorical_features = ['merchant', 'category', 'job', 'age_group']
    X[categorical_features] = X[categorical_features].astype('category')

    for col in categorical_features:
        le = LabelEncoder()
        # Fit on the category columns for encoding
        X[col] = le.fit_transform(X[col])
    
    #print(df.info())

    return X, y

def train_and_evaluate(X, y):
    """
    Trains and evaluates four models, returning the best one.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1, device='gpu', gpu_platform_id=0, gpu_device_id=0),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1, tree_method='gpu_hist'),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42, task_type='GPU', devices='0'),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1)
    }

    best_model_name = None
    best_f1 = 0
    best_recall = 0

    print("Starting model training and evaluation...")
    for name, model in models.items():
        print(f"Training {name}...")
        
        # # Check for GPU support and handle accordingly
        # if name in ['LightGBM', 'XGBoost', 'CatBoost']:
        #     model.fit(X_train, y_train) # No scaling needed for tree-based models
        #     y_pred = model.predict(X_test)
        # else: # RandomForest requires scaled data for consistency, though it's not sensitive to it
        #     model.fit(X_train_scaled, y_train)
        #     y_pred = model.predict(X_test_scaled)
        # Fit the model
        with tqdm(total=1, desc="Training") as pbar:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            pbar.update(1)
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"--- {name} Results ---")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Contusion Matrix:\n", conf_matrix)

        # Select the best model based on F1 score and Recall
        if f1 > best_f1 and recall > best_recall:
            best_f1 = f1
            best_recall = recall
            best_model_name = name
            best_model = model
    
    print(f"The best performing model is: {best_model_name} with F1: {best_f1:.4f} and Recall: {best_recall:.4f}")
    
    # Save the best model
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    joblib.dump(best_model, f'checkpoints/{best_model_name.lower()}_model.joblib')
    print(f"Best model saved to checkpoints/{best_model_name.lower()}_model.joblib")
    
    # Also save the scaler for the RandomForest model if it's the best one
    if best_model_name == 'RandomForest':
        joblib.dump(scaler, 'checkpoints/scaler.joblib')
        print("Scaler for RandomForest model also saved.")

if __name__ == "__main__":
    df = pd.read_csv('data/fraudTrain.csv')
    X, y = preprocess_data(df)
    print(X.info())

    
    train_and_evaluate(X, y)