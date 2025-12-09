import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from src.shared.load_data import load_all_datasets
from src.shared.extract_features import extract_features


def main():

    print("Loading dataset...")
    df = load_all_datasets()
    print("Total samples:", len(df))

    print("\nExtracting features...")
    X, y, groups = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        X.append(extract_features(row["path"]))
        y.append(row["emotion"])
        groups.append(f"{row['dataset']}_{row['actor']}")

    X = np.array(X)
    groups = np.array(groups)

    print("\nEncoding labels...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("\nSplitting dataset (speaker-independent)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nStarting Random Forest Grid Search (this may take a few minutes)...")

    param_grid = {
        "n_estimators": [300, 600],
        "max_depth": [30, 50, None],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
        "bootstrap": [True]
    }

    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid.fit(X_train_scaled, y_train)

    print("\nBest RF parameters found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    print("\nEvaluating model on held-out test set...")
    y_pred = best_model.predict(X_test_scaled)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

    print("\nSaving the trained model...")
    joblib.dump(best_model, "models/all-rf/rf_model.pkl")
    joblib.dump(scaler, "models/all-rf/rf_scaler.pkl")
    joblib.dump(encoder, "models/all-rf/rf_label_encoder.pkl")

    print("\nDONE! Optimized RF model trained and saved.")


if __name__ == "__main__":
    main()
