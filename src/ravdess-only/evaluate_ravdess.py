import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import GroupShuffleSplit

from src.shared.load_data import load_ravdess
from src.shared.extract_features import extract_features


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


def main():

    print_section("STEP 1 — Loading RAVDESS Dataset")
    df = load_ravdess("./data/ravdess")
    print(f"Total samples: {len(df)}")

    print_section("STEP 2 — Extracting Features")
    X, y, groups = [], [], []

    for _, row in df.iterrows():
        feat = extract_features(row["path"])
        X.append(feat)
        y.append(row["emotion"])
        groups.append(row["actor"])

    X = np.array(X)
    groups = np.array(groups)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print_section("STEP 3 — Loading Saved Model Files")
    model = joblib.load("models/ravdess/svm_model_ravdess.pkl")
    scaler = joblib.load("models/ravdess/scaler_ravdess.pkl")
    encoder = joblib.load("models/ravdess/label_encoder_ravdess.pkl")

    y_encoded = encoder.transform(y)

    print_section("STEP 4 — Speaker-Independent Split")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    X_test = X[test_idx]
    y_test = y_encoded[test_idx]

    print(f"Test samples: {len(y_test)}")

    print_section("STEP 5 — Scaling Test Features")

    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print_section("STEP 6 — Predicting")
    y_pred = model.predict(X_test_scaled)

    print_section("STEP 7 — Evaluation Metrics")

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro):   {precision_macro:.4f}")
    print(f"Recall (Macro):      {recall_macro:.4f}")
    print(f"F1-Score (Macro):    {f1_macro:.4f}")
    print(f"Precision (Weighted): {precision_weighted:.4f}")
    print(f"Recall (Weighted):    {recall_weighted:.4f}")
    print(f"F1-Score (Weighted):  {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n====================== DONE ======================\n")


if __name__ == "__main__":
    main()
