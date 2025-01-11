
import Sentiment_Analysis
from Sentiment_Analysis import *

# Initialize MLflow experiment
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import RidgeClassifier
mlflow.set_experiment("Text Classification Experiment")
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Define the pipeline
pipe = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', RidgeClassifier())
])

# Train the pipeline
pipe.fit(X_train, y_train)

# Make predictions
y_pred = pipe.predict(X_test)

# Calculate overall metrics
accuracy = accuracy_score(y_test, y_pred)
weighted_precision = precision_score(y_test, y_pred, average='weighted')
weighted_recall = recall_score(y_test, y_pred, average='weighted')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

# Calculate class-wise metrics
classwise_metrics = classification_report(y_test, y_pred, output_dict=True)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1 Score: {weighted_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Start an MLflow run
with mlflow.start_run() as run:
    # Log overall metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("weighted_precision", weighted_precision)
    mlflow.log_metric("weighted_recall", weighted_recall)
    mlflow.log_metric("weighted_f1", weighted_f1)

    # Log class-wise metrics
    for label, metrics in classwise_metrics.items():
        if isinstance(metrics, dict):  # Skip overall metrics like 'accuracy'
            mlflow.log_metric(f"{label}_precision", metrics['precision'])
            mlflow.log_metric(f"{label}_recall", metrics['recall'])
            mlflow.log_metric(f"{label}_f1_score", metrics['f1-score'])

    # Generate and log confusion matrix as an artifact
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    conf_matrix_path = "confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    mlflow.log_artifact(conf_matrix_path)

    # Log the model with input example and signature
    input_example = {"Lemmatizer_data": ["sample text for testing"]}
    signature = mlflow.models.infer_signature(X_train[:1], pipe.predict(X_train[:1]))

    mlflow.sklearn.log_model(
        sk_model=pipe,
        artifact_path="ridge_pipeline",
        registered_model_name="RidgeClassifierPipeline",
        signature=signature,
        input_example=input_example
    )

    print(f"Model logged in MLflow Run: {run.info.run_id}")

