# MLflow-Tracker
# 📚 NLP Project with MLflow - Step-by-Step Guide

## 🚀 Project Overview
This project demonstrates how to process text data, train machine learning models for classification, and track experiments using MLflow. Follow the steps below to get started!

---

## 🛠️ Setup Instructions

### 1️⃣ Install Prerequisites
Ensure Python is installed (≥3.8 recommended). Install required libraries using the `requirements.txt` file:

     ```bash
     pip install -r requirements.txt

## 2️⃣ Prepare Your Data

Ensure you have your training (`Train.csv`) and validation (`Valid.csv`) datasets in the proper format. Example structure:
Each dataset should have:

- **text**: The raw text input.
- **label**: The target label (e.g., `0` or `1` for binary classification).

## 🔍 Text Preprocessing Steps

The text undergoes multiple preprocessing steps, including:

1. **Removing Stopwords**: Filters out common words like "the" and "is."
2. **Removing URLs, Hashtags, and Mentions**.
3. **Punctuation Removal**: Deletes unnecessary symbols.
4. **Stemming & Lemmatization**: Reduces words to their root forms.

These steps help clean the data and prepare it for model training.
## 🤖 Model Training

### 3️⃣ Train a Classifier

We use Scikit-learn's pipeline to train models. Example of a Ridge Classifier:

     ```python
     from sklearn.linear_model import RidgeClassifier

# Train and evaluate the pipeline
test_pipeline(RidgeClassifier())

## 4️⃣ Log Metrics and Models with MLflow

To track experiments and log models, **MLflow** is integrated into the training process:

     ```python
     import mlflow
     import mlflow.sklearn
     with mlflow.start_run():
         # Log metrics like accuracy, precision, recall, and F1-score
         mlflow.log_metric("accuracy", accuracy_score)
         mlflow.log_metric("precision", precision_score)
         mlflow.log_metric("recall", recall_score)
         mlflow.log_metric("f1_score", f1_score)
         # Save the trained model
         mlflow.sklearn.log_model(pipe, "model")
## 📈 Tracking Experiments

### 5️⃣ Run MLflow UI
Launch the MLflow UI to visualize experiments, metrics, and models:

     ```bash
     mlflow ui

Access the UI in your browser at: http://127.0.0.1:5000.

## 🗂️ Key Features

### ✅ Metrics Tracking
- Logs overall and class-specific metrics, including accuracy, precision, recall, and F1-score.
- Generates and saves a confusion matrix as an artifact.

### ✅ Model Logging and Reusability
- Save and register trained models using:

     ```python
     mlflow.sklearn.log_model(pipe, "model")
     ```python
     loaded_model = mlflow.sklearn.load_model("runs:/<run_id>/model")


## 📂 How to Use

### 🔨 Steps to Run

1. **Clone this repository:**

     ```bash
     git clone <repository_url>
     cd <repository_folder>
2. **Install dependencies:**

     ```bash
     pip install -r requirements.txt
3. **Train the model::**

     ```bash
     python Mlfow.py
4. **Start MLflow UI for tracking:**

     ```bash
     mlflow ui
Access your results at http://127.0.0.1:5000.

## 🔗 Outputs

- **Trained Model**: Saved in MLflow under `artifacts/model`.
- **Confusion Matrix**: Logged as an artifact for visualization.
- **Metrics**: Accuracy, precision, recall, and F1-score for each experiment.





