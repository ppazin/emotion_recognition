import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from load_data import load_all_datasets
from extract_features import extract_features

def main():
    print("Loading data...")
    df = load_all_datasets()

    X = []
    y = []
    groups = []

    for _, row in df.iterrows():
        X.append(extract_features(row["path"]))
        y.append(row["emotion"])
        groups.append(f"{row['dataset']}_{row['actor']}")

    X = np.array(X)

    model = joblib.load("models/svm_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/label_encoder.pkl")

    y_enc = encoder.transform(y)

    print("Group-wise split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))

    X_test = X[test_idx]
    y_test = y_enc[test_idx]

    X_test_scaled = scaler.transform(X_test)

    print("Predicting...")
    y_pred = model.predict(X_test_scaled)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
