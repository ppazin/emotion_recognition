import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from emotion_recognition.src.shared.load_data import load_ravdess
from emotion_recognition.src.shared.extract_features import extract_features


def main():
    print("Loading RAVDESS dataset...")
    df = load_ravdess("./data/ravdess/audio_speech_actors_01-24")
    print("Total samples:", len(df))

    print("\nExtracting features...")
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

    print("\nEncoding labels...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("\nSplitting dataset (speaker independent)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    print("Training complete.")

    print("\nSaving model files to /models ...")
    joblib.dump(model, "models/ravdess-rf/rf_model_ravdess.pkl")
    joblib.dump(scaler, "models/ravdess-rf/rf_scaler_ravdess.pkl")
    joblib.dump(encoder, "models/ravdess-rf/rf_label_encoder_ravdess.pkl")

    print("\nDONE! Random Forest model saved successfully.")


if __name__ == "__main__":
    main()
