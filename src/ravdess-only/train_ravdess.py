import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm

from src.shared.load_data import load_ravdess
from src.shared.extract_features import extract_features


def main():
    print("Loading RAVDESS dataset...")
    df = load_ravdess("./data/ravdess")
    print("Total samples:", len(df))

    print("\nExtracting features (MFCC + others)... this may take a few minutes")

    X = []
    y = []
    groups = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        features = extract_features(row["path"])
        X.append(features)
        y.append(row["emotion"])
        groups.append(row["actor"])

    X = np.array(X)
    groups = np.array(groups)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nSplitting dataset (train/test) with GroupShuffleSplit (speaker independent)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    print("\nScaling features...")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTraining SVM with GridSearchCV...")

    params = {
        "C": [0.1, 1, 10, 100, 1000],
        "gamma": ["scale", "auto", 0.1, 0.01, 0.001],
        "kernel": ["rbf"],
    }

    svc = SVC(class_weight="balanced")

    grid = GridSearchCV(
        svc,
        params,
        cv=3,
        n_jobs=-1,
        verbose=1,
        error_score="raise"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("\nBest model:", best_model)

    print("\nSaving model files into /models ...")

    joblib.dump(best_model, "models/ravdess/svm_model_ravdess.pkl")
    joblib.dump(scaler, "models/ravdess/scaler_ravdess.pkl")
    joblib.dump(label_encoder, "models/ravdess/label_encoder_ravdess.pkl")

    print("\nDONE! Model, scaler, and label encoder saved successfully.")


if __name__ == "__main__":
    main()
