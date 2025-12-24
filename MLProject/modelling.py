import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from dotenv import load_dotenv
load_dotenv()

def load_data(input_dir):
    train_path = os.path.join(input_dir, 'train.csv')
    test_path = os.path.join(input_dir, 'test.csv')
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    target_col = 'Heart Disease Status'
    
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]
    
    return X_train, y_train, X_test, y_test

def train_and_log(input_dir):
    # Setup DagsHub & MLflow
    dagshub.init(repo_owner='rafli21xrplc', repo_name='asah-ml-flow', mlflow=True)
    mlflow.set_experiment("Heart Disease Experiment")

    with mlflow.start_run(run_name="RandomForest_Baseline") as run:
        run_id = run.info.run_id
        print(f"Current Run ID: {run_id}")
        
        print("Loading data...")
        X_train, y_train, X_test, y_test = load_data(input_dir)
        
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "RandomForest")

        print("Training model...")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(model, "model")
        
        local_model_path = "model_lokal_output"
        
        # Hapus folder lama jika ada (agar tidak error)
        if os.path.exists(local_model_path):
            import shutil
            shutil.rmtree(local_model_path)
            
        print(f"Saving model to local folder: {local_model_path}...")
        mlflow.sklearn.save_model(model, local_model_path)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc}")
        
        mlflow.log_metric("accuracy", acc)
        
        print("Generating confusion matrix...")
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        with open("last_run_id.txt", "w") as f:
            f.write(run_id)
        
        if os.path.exists(cm_path):
            os.remove(cm_path)
            
        print(f"Run ID {run_id} saved. Training Selesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='heart_disease_preprocessing', help='Input directory')
    args = parser.parse_args()
    
    train_and_log(args.input)