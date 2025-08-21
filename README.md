# Credit Card Fraud Detection Project

This project implements a credit card fraud detection system using various machine learning models. It features a complete end-to-end workflow, including data preprocessing, model training, evaluation, and a user-friendly GUI for real-time fraud prediction on new data.

## 🚀 Features

- **Four Models:** Trains and compares LightGBM, XGBoost, CatBoost, and RandomForest classifiers.
- **GPU Acceleration:** Utilizes GPU acceleration for supported models to speed up the training process.
- **Robust Preprocessing:** Follows a specific data preprocessing and feature engineering pipeline to prepare the data for modeling.
- **Interactive GUI:** A modern, clean user interface built with `customtkinter` for easy test file selection and prediction.
- **Model Persistence:** Automatically saves the best-performing model (based on F1 score and Recall) for later use.

## 📁 Project Structure

credit-card-fraud-detection/
├── data/
│   ├── fraud_subset_1.csv         # Training data
│   └── fraudTest_subset_1.csv     # Test data
├── checkpoints/
├── .gitignore
├── README.md
├── requirements.txt
├── train.py
├── demo.py

## 📋 Getting Started

### Prerequisites

-   Python 3.8+
-   A system with an NVIDIA GPU for accelerated training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your_repo_url>
    cd credit-card-fraud-detection
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    -   On Windows: `venv\Scripts\activate`
    -   On macOS/Linux: `source venv/bin/activate`
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Note: The GPU-accelerated versions of LightGBM, XGBoost, and CatBoost will be installed.

### Data

Download the `fraud_subset_1.csv` (for training) and `fraudTest_subset_1.csv` (for testing) files and place them in the `data/` directory.

## 🏃 Running the Project

### Step 1: Model Training

Run the training script to preprocess the data, train the models, and save the best-performing one to the `checkpoints/` folder.

python train.py

### Step 2: Running the Prediction Demo

After the training is complete, run the demo script to use the saved model for predictions on new data. A GUI window will appear.

python demo.py