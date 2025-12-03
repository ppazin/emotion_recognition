import os
import sys
import tempfile

import numpy as np
import streamlit as st
import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append("src")

from emotion_recognition.src.shared.extract_features import extract_features
from emotion_recognition.src.shared.load_data import load_all_datasets, load_ravdess


MODEL_CONFIG = {
    "combined": {
        "label": "SVM – RAVDESS + CREMA-D (8 emocija)",
        "model_path": "models/svm_model.pkl",
        "scaler_path": "models/scaler.pkl",
        "encoder_path": "models/label_encoder.pkl",
        "dataset": "all",
    },
    "ravdess": {
        "label": "SVM – samo RAVDESS",
        "model_path": "models/svm_model_ravdess.pkl",
        "scaler_path": "models/scaler_ravdess.pkl",
        "encoder_path": "models/label_encoder_ravdess.pkl",
        "dataset": "ravdess",
    },
}


@st.cache_resource
def load_model(model_key: str):
    cfg = MODEL_CONFIG[model_key]
    model = joblib.load(cfg["model_path"])
    scaler = joblib.load(cfg["scaler_path"])
    label_encoder = joblib.load(cfg["encoder_path"])
    return model, scaler, label_encoder



def predict_emotion_from_file(file_bytes: bytes, filename: str, model_key: str):
    _, ext = os.path.splitext(filename)
    if ext == "":
        ext = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path)
        model, scaler, encoder = load_model(model_key)
        features_scaled = scaler.transform([features])
        y_pred = model.predict(features_scaled)
        emotion = encoder.inverse_transform(y_pred)[0]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return emotion



@st.cache_data(show_spinner=False)
def evaluate_model(model_key: str):
    cfg = MODEL_CONFIG[model_key]
    model, scaler, encoder = load_model(model_key)

    if cfg["dataset"] == "ravdess":
        df = load_ravdess("./data/ravdess")
    else:
        df = load_all_datasets()

    X = []
    y = []
    groups = []

    for _, row in df.iterrows():
        X.append(extract_features(row["path"]))
        y.append(row["emotion"])
        groups.append(f"{row['dataset']}_{row['actor']}")

    X = np.array(X)
    y_enc = encoder.transform(y)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))

    X_test = X[test_idx]
    y_test = y_enc[test_idx]

    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=encoder.classes_
    )
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)

    return acc, report, cm_df



def init_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def add_message(role, text):
    st.session_state["messages"].append({"role": role, "text": text})



def render_chat_page(model_key: str):
    st.subheader("Razgovor")

    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["text"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["text"])

    st.divider()
    st.subheader("Pošalji novi audio")

    uploaded_file = st.file_uploader(
        "Odaberi audio datoteku (wav/mp3/ogg/flac)...",
        type=["wav", "mp3", "ogg", "flac"],
    )

    user_note = st.text_input(
        "Opcionalna poruka uz audio (npr. što govoriš ili kako se osjećaš):",
        value="",
        placeholder="Npr. 'pričam veselo na engleskom'...",
    )

    analyze_clicked = st.button("Analiziraj")

    if analyze_clicked:
        if uploaded_file is None:
            st.warning("Najprije uploadaj audio datoteku.")
        else:
            user_msg_text = f"📎 Poslao sam audio: **{uploaded_file.name}**"
            if user_note.strip():
                user_msg_text += f"\n\n Napomena: {user_note}"

            add_message("user", user_msg_text)

            with st.spinner("Analiziram emociju iz glasa..."):
                file_bytes = uploaded_file.read()
                try:
                    predicted_emotion = predict_emotion_from_file(
                        file_bytes, uploaded_file.name, model_key
                    )
                    bot_text = f"Prepoznata emocija: **{predicted_emotion}**"
                except Exception as e:
                    bot_text = f"Greška pri obradi audio datoteke: `{e}`"
            add_message("assistant", bot_text)

            st.rerun()


def render_info_page(model_key: str):
    cfg = MODEL_CONFIG[model_key]

    st.subheader("Informacije o sustavu")

    st.markdown(
        f"""
    ### Trenutno odabrani model

    **{cfg['label']}**

    - Klasični strojno-učeći model (SVM)
    - Ručno dizajnirane značajke:
    - MFCC, delta, delta-delta  
    - Chroma  
    - RMS (energija)  
    - Spectral contrast  
    - Trening skup:
    - `{cfg['dataset']}` (ovisno o modelu, npr. samo RAVDESS ili RAVDESS+CREMA-D)

    ---

    ### Napomena

    - Model je treniran na **engl. govoru glumaca**, u kontroliranim uvjetima.
    - Za **hrvatski** i svakodnevni govor rezultati će biti slabiji – to je odličan primjer ograničenja
    klasičnih ML metoda i domenskog prijenosa.
    - U radu možeš ovo iskoristiti za:
    - usporedbu različitih datasetova,
    - usporedbu različitih modela,
    - diskusiju o generalizaciji na nove govornike.
    """
    )


def render_eval_page(model_key: str):
    cfg = MODEL_CONFIG[model_key]
    st.subheader("Evaluacija modela")

    st.markdown(
        f"""
        Ovdje možeš pokrenuti evaluaciju za trenutno odabrani model:

        **{cfg['label']}**

        Evaluacija koristi:
        - **GroupShuffleSplit** (govornik-nezavisna podjela)
        - **test_size = 0.2**
        - metrika: *accuracy*, *precision/recall/F1* po klasi i matrica zabune.
        """
        )

    if st.button("Pokreni evaluaciju"):
        with st.spinner("Izračunavam metrike, ovo može potrajati..."):
            acc, report, cm_df = evaluate_model(model_key)

        st.success(f"Gotovo! Accuracy: **{acc:.3f}**")

        st.markdown("#### Classification report")
        st.text(report)

        st.markdown("#### Confusion matrix")
        st.dataframe(cm_df)



def main():
    st.set_page_config(page_title="Pamonia")
    st.title("Pamonia – Prepoznavanje emocija iz glasa")

    init_session_state()

    with st.sidebar:
        st.header("Navigacija")

        page = st.radio(
            "Odaberi sekciju:",
            ["Chat", "Info", "Evaluacija"],
        )

        st.markdown("---")
        st.subheader("Model")

        model_key = st.selectbox(
            "Odaberi model:",
            options=list(MODEL_CONFIG.keys()),
            format_func=lambda k: MODEL_CONFIG[k]["label"],
        )


    if page == "Chat":
        render_chat_page(model_key)
    elif page == "Info":
        render_info_page(model_key)
    elif page == "Evaluacija":
        render_eval_page(model_key)


if __name__ == "__main__":
    main()
