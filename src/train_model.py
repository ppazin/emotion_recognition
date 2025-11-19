import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm

from load_data import load_all_datasets
from extract_features import extract_features


def main():
    print("Loading dataset...")
    df = load_all_datasets()
    print("Total samples:", len(df))

    print("\n Extracting features (MFCC + others)... this may take a few minutes")

    X = []
    y = []
    groups = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        features = extract_features(row["path"])
        X.append(features)
        y.append(row["emotion"])
        groups.append(f"{row['dataset']}_{row['actor']}")

    X = np.array(X)
    groups = np.array(groups)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\n Splitting dataset (train/test) with GroupShuffleSplit...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    print("\n Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\n Training SVM with GridSearchCV...")
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
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("\nBest model:", best_model)

    print("\nSaving model files into /models ...")

    joblib.dump(best_model, "models/svm_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    print("\nDONE! Model, scaler, and label encoder saved successfully.")


if __name__ == "__main__":
    main()
