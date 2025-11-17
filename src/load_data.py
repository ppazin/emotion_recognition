import os
import glob
import pandas as pd

ravdess_emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def load_ravdess(base_path):
    audio_files = glob.glob(os.path.join(base_path, "**/*.wav"), recursive=True)

    data = []

    for file in audio_files:
        file_name = os.path.basename(file)
        parts = file_name.split("-")

        emotion_id = parts[2]
        actor_id = parts[-1].split(".")[0]

        emotion = ravdess_emotions.get(emotion_id)

        data.append([file, emotion, actor_id, "RAVDESS"])

    df = pd.DataFrame(data, columns=["path", "emotion", "actor", "dataset"])
    return df

cremad_emotions = {
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEAR": "fearful",
    "DIS": "disgust",
    "NEU": "neutral"
}

def load_cremad(base_path):
    audio_files = glob.glob(os.path.join(base_path, "*.wav"))

    data = []

    for file in audio_files:
        file_name = os.path.basename(file)
        
        parts = file_name.split("_")
        actor_id = parts[0]
        emotion_code = parts[2].split(".")[0]

        emotion = cremad_emotions.get(emotion_code)

        data.append([file, emotion, actor_id, "CREMA-D"])

    df = pd.DataFrame(data, columns=["path", "emotion", "actor", "dataset"])
    return df

def load_all_datasets():
    ravdess_df = load_ravdess("./data/ravdess")
    cremad_df = load_cremad("./data/cremad/AudioWAV")

    df = pd.concat([ravdess_df, cremad_df], ignore_index=True)
    df = df.dropna(subset=["emotion"])
    return df


def main():
    df = load_all_datasets()
    print("Ukupno audio zapisa učitano:", len(df))
    print(df["dataset"].value_counts())
    print(df.head())
if __name__ == "__main__":
    main()

