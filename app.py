import os
import sys
import tempfile

import numpy as np
import streamlit as st
import joblib

sys.path.append("src")

from src.extract_features import extract_features


@st.cache_resource
def load_model():
    model = joblib.load("models/svm_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, scaler, label_encoder


def predict_emotion_from_file(file_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path)
        model, scaler, encoder = load_model()
        features_scaled = scaler.transform([features])
        y_pred = model.predict(features_scaled)
        emotion = encoder.inverse_transform(y_pred)[0]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return emotion


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def add_message(role, text):
    st.session_state["messages"].append({"role": role, "text": text})


def main():
    st.set_page_config(page_title="Pamonia")
    st.title("Pamonia")

    st.write(
        "Uploadaj **audio (.wav)**, a model će pokušati prepoznati emociju "
        "na temelju tvog treniranog SVM + MFCC sustava."
    )

    init_session_state()
    model, scaler, encoder = load_model()

    with st.sidebar:
        st.header("Informacije o modelu")
        st.markdown(
            """
        **Model:**
        - Klasični ML (SVM)
        - Značajke: MFCC, delta, delta-delta, chroma, RMS, spectral contrast  
        - Trenirano na RAVDESS + CREMA-D

        **Napomena:**
        - Rezultati možda nisu 100% točni, ovisno o kvaliteti audio zapisa i izraženosti emocije
        - Model je treniran na engleskom govoru, pa rezultati za druge jezike mogu varirati
        """
        )

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
        "Odaberi .wav datoteku",
        type=["wav"],
    )


    analyze_clicked = st.button("Analiziraj")

    if analyze_clicked:
        if uploaded_file is None:
            st.warning("Najprije uploadaj .wav datoteku.")
        else:
            user_msg_text = f"📎 Poslao sam audio: **{uploaded_file.name}**"

            add_message("user", user_msg_text)

            with st.spinner("Analiziram emociju iz glasa..."):
                file_bytes = uploaded_file.read()
                try:
                    predicted_emotion = predict_emotion_from_file(file_bytes, uploaded_file.name)
                    bot_text = f"Prepoznata emocija: **{predicted_emotion}**"
                except Exception as e:
                    bot_text = f"Greška pri obradi audio datoteke: `{e}`"

            add_message("assistant", bot_text)

            st.rerun()


if __name__ == "__main__":
    main()
