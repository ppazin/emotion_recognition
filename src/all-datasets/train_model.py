import numpy as np
import joblib
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier

from src.shared.load_data import load_all_datasets
from src.shared.extract_features import extract_features


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


def main():

    print_section("STEP 1 — Loading Dataset")
    df = load_all_datasets()
    print(f"Total samples found: {len(df)}")

    print_section("STEP 2 — Extracting Features")
    X, y, groups = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        features = extract_features(row["path"])
        X.append(features)
        y.append(row["emotion"])
        groups.append(f"{row['dataset']}_{row['actor']}")

    X = np.array(X)
    groups = np.array(groups)

    print_section("STEP 3 — Encoding Labels")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    print("Labels encoded:", list(encoder.classes_))

    print_section("STEP 4 — Speaker-Independent Split")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    print_section("STEP 5 — Scaling Features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print_section("STEP 6 — Training XGBoost Model")
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(y_train)),
        learning_rate=0.05,
        n_estimators=600,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    print_section("STEP 7 — Evaluating Model")
    y_pred = model.predict(X_test_scaled)

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

    print_section("STEP 8 — Saving Final Model")
    joblib.dump(model, "models/all-rf/model.pkl")
    joblib.dump(scaler, "models/all-rf/scaler.pkl")
    joblib.dump(encoder, "models/all-rf/label_encoder.pkl")

    print("Model, scaler, and label encoder saved successfully.")
    print("\n====================== DONE ======================")


if __name__ == "__main__":
    main()
