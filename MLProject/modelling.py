import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fungsi load data tetap sama
def load_data(input_dir):
    train = pd.read_csv(f"{input_dir}/train.csv")
    test = pd.read_csv(f"{input_dir}/test.csv")
    X_train = train.drop(columns=['Heart Disease Status'])
    y_train = train['Heart Disease Status']
    X_test = test.drop(columns=['Heart Disease Status'])
    y_test = test['Heart Disease Status']
    return X_train, y_train, X_test, y_test

def train(input_dir):
    # 1. Aktifkan Autolog (Ini kuncinya!)
    # Autolog akan otomatis merekam:
    # - Metrik (accuracy, precision, dll)
    # - Parameter (n_estimators, max_depth, dll)
    # - Artefak model (termasuk estimator.html)
    mlflow.sklearn.autolog()

    # 2. Start Run (Tanpa koneksi ke DagsHub)
    with mlflow.start_run(run_name="Basic_Autolog_Run"):
        X_train, y_train, X_test, y_test = load_data(input_dir)
        
        # 3. Training Model
        # Tidak perlu lagi mlflow.log_param atau log_model manual
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi (opsional, autolog biasanya sudah menghitung metrik dasar)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc}")
        
        # Simpan Run ID untuk keperluan Docker (Kriteria 3)
        run_id = mlflow.active_run().info.run_id
        with open("last_run_id.txt", "w") as f:
            f.write(run_id)
            
        print(f"Training selesai. Run ID: {run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='heart_disease_preprocessing')
    args = parser.parse_args()
    train(args.input)