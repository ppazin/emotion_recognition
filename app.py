import os
import sys
import tempfile

import streamlit as st
import joblib

sys.path.append("src")
from src.shared.extract_features import extract_features


MODEL_PATH = "models/all-rf/model.pkl"
SCALER_PATH = "models/all-rf/scaler.pkl"
ENCODER_PATH = "models/all-rf/label_encoder.pkl"

EMOTION_STYLES = {
    "neutral":   {"label": "Neutralno",   "color": "#8c8c8c"},
    "calm":      {"label": "Smireno",    "color": "#52c41a"},
    "happy":     {"label": "Sretno",     "color": "#faad14"},
    "sad":       {"label": "Tužno",      "color": "#40a9ff"},
    "angry":     {"label": "Ljuto",    "color": "#ff4d4f"},
    "fearful":   {"label": "Uplašeno",   "color": "#722ed1"},
    "disgust":   {"label": "Gadljivo",   "color": "#13c2c2"},
    "surprised": {"label": "Iznenađeno", "color": "#eb2f96"},
}


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, scaler, label_encoder


def predict_emotion_from_file(file_bytes: bytes, filename: str) -> str:
    _, ext = os.path.splitext(filename)
    if ext == "":
        ext = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
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


def render_global_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #1f2933 0, #0b1015 55%, #02040a 100%);
            color: #f5f5f5;
        }

        h1, h2, h3, h4 {
            letter-spacing: 0.03em;
        }

        .app-header {
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .app-title {
            margin-bottom: 0.2rem;
            font-size: 2.2rem;
            font-weight: 700;
        }

        .app-subtitle {
            color: #a3a3a3;
            font-size: 0.95rem;
            margin: 0;
        }

        .card {
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.28);
            backdrop-filter: blur(10px);
        }

        .card-subtle {
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.18);
            backdrop-filter: blur(8px);
            margin-bottom: 1rem;
        }

        div[data-testid="stFileUploader"] > section {
            border-radius: 14px !important;
            border: 1px dashed rgba(148, 163, 184, 0.6) !important;
            background-color: rgba(15, 23, 42, 0.55) !important;
            padding: 1.4rem 1.6rem !important;
        }

        div[data-testid="stFileUploader"] label {
            font-size: 0.9rem;
            color: #cbd5f5 !important;
        }

        .result-header {
            font-size: 0.9rem;
            color: #9ca3af;
            margin-bottom: 0.15rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        .result-emotion {
            font-size: 1.7rem;
            font-weight: 600;
        }

        .emotion-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.45rem;
        }

        /* Small section title on left */
        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Pamonia – Emocije iz glasa",
        layout="wide",
    )

    render_global_styles()

    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">Pamonia</div>
            <p class="app-subtitle">
                Prepoznavanje emocija iz glasa pomoću klasičnih modela strojnog učenja (Random Forest + XGBoost)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.25, 1])

    with col_left:
        st.markdown("### Učitaj audio")

        st.markdown(
            """
            <div class="card-subtle">
                <p style="font-size: 0.9rem; color: #e5e7eb; margin-bottom: 0.4rem;">
                    Ispusti audio datoteku unutar ovog područja ili klikni za odabir.
                </p>
                <p style="font-size: 0.8rem; color: #9ca3af; margin-bottom: 0;">
                    Podržani formati: <code>wav</code>, <code>mp3</code>, <code>ogg</code>, <code>flac</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Ispusti audio datoteku ovdje ili klikni za odabir",
            type=["wav", "mp3", "ogg", "flac"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            with st.spinner("Analiziram emociju iz glasa..."):
                try:
                    file_bytes = uploaded_file.read()
                    emotion_raw = predict_emotion_from_file(file_bytes, uploaded_file.name)
                    emotion_key = str(emotion_raw).lower()
                    style = EMOTION_STYLES.get(emotion_key)

                    st.success("Analiza završena.")

                    if style:
                        label = style["label"]
                        color = style["color"]

                        st.markdown(
                            f"""
                            <div class="card" style="margin-top: 0.9rem;">
                                <div class="result-header">Prepoznata emocija</div>
                                <div class="result-emotion">{label}</div>
                                <div class="emotion-pill" style="border: 1px solid {color}; color: {color};">
                                    {emotion_raw}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="card" style="margin-top: 0.9rem;">
                                <div class="result-header">Prepoznata emocija</div>
                                <div class="result-emotion">{emotion_raw}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                except Exception as e:
                    st.error(f"Greška pri obradi audio datoteke: `{e}`")
        else:
            st.info("Još nije učitan nijedan zapis. Ispusti datoteku ili klikni za odabir.")

    with col_right:
        st.markdown("### Informacije o modelu")

        st.markdown(
            """
            <div class="card">
                <p style="margin-bottom: 0.6rem;">
                    <strong>Model</strong>
                </p>
                <ul style="margin-top: 0; padding-left: 1.1rem; font-size: 0.9rem;">
                    <li>Korištene metode strojnog učenja: XGBoost</li>
                    <li>Analizirane značajke:
                        <ul style="margin-top: 0.25rem; padding-left: 1.1rem;">
                            <li>MFCC, delta, delta-delta</li>
                            <li>Chroma</li>
                            <li>RMS (energija)</li>
                            <li>Spectral contrast</li>
                        </ul>
                    </li>
                </ul>
                <p style="margin-bottom: 0.4rem; margin-top: 0.9rem;">
                    <strong>Datasetovi</strong>
                </p>
                <ul style="margin-top: 0; padding-left: 1.1rem; font-size: 0.9rem;">
                    <li>RAVDESS</li>
                    <li>TESS</li>
                    <li>CREMA-D</li>
                    <li>SAVEE</li>
                </ul>
                <p style="margin-bottom: 0.4rem; margin-top: 0.9rem;">
                    <strong>Napomene</strong>
                </p>
                <ul style="margin-top: 0; padding-left: 1.1rem; font-size: 0.9rem; color: #d1d5db;">
                    <li>Model je treniran na glumcima i snimkama na engleskom jeziku.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
