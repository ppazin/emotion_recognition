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
    "08": "surprised",
}


def load_ravdess(base_path):
    audio_files = glob.glob(os.path.join(base_path, "**/*.wav"), recursive=True)

    data = []

    for file in audio_files:
        if "audio_speech_actors" not in file.lower():
            continue

        file_name = os.path.basename(file)
        parts = file_name.split("-")

        emotion_id = parts[2]
        actor_id = parts[-1].split(".")[0]

        emotion = ravdess_emotions.get(emotion_id)

        data.append([file, emotion, actor_id, "RAVDESS"])

    return pd.DataFrame(data, columns=["path", "emotion", "actor", "dataset"])


cremad_emotions = {
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEAR": "fearful",
    "DIS": "disgust",
    "NEU": "neutral",
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


tess_emotions = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "pleasant_surprise": "surprised",
    "pleasant_surprised": "surprised",
    "sad": "sad",
}


def load_tess(base_path):
    audio_files = glob.glob(os.path.join(base_path, "**/*.wav"), recursive=True)

    data = []

    for file in audio_files:
        parent = os.path.basename(os.path.dirname(file))
        parts = parent.split("_", 1)

        actor_id = parts[0]

        if len(parts) > 1:
            emotion_key = parts[1].lower()
        else:
            emotion_key = ""

        emotion = tess_emotions.get(emotion_key)

        data.append([file, emotion, actor_id, "TESS"])

    df = pd.DataFrame(data, columns=["path", "emotion", "actor", "dataset"])
    return df

savee_emotions = {
    "a": "angry",
    "d": "disgust",
    "f": "fearful",
    "h": "happy",
    "n": "neutral",
    "sa": "sad",
    "su": "surprised",
}


def load_savee(base_path):
    audio_files = glob.glob(os.path.join(base_path, "*.wav"))

    data = []

    for file in audio_files:
        file_name = os.path.basename(file)
        name_no_ext = os.path.splitext(file_name)[0]

        parts = name_no_ext.split("_")
        actor_id = parts[0]

        if len(parts) > 1:
            emo_part = parts[1]
        else:
            emo_part = ""

        emotion_code = "".join(ch for ch in emo_part if ch.isalpha())
        emotion = savee_emotions.get(emotion_code)

        data.append([file, emotion, actor_id, "SAVEE"])

    df = pd.DataFrame(data, columns=["path", "emotion", "actor", "dataset"])
    return df



def load_all_datasets():
    dfs = []

    if os.path.isdir("./data/ravdess"):
        dfs.append(load_ravdess("./data/ravdess/audio_speech_actors_01-24"))

    if os.path.isdir("./data/cremad/AudioWAV"):
        dfs.append(load_cremad("./data/cremad/AudioWAV"))

    if os.path.isdir("./data/tess"):
        dfs.append(load_tess("./data/tess"))

    if os.path.isdir("./data/savee"):
        dfs.append(load_savee("./data/savee"))

    if not dfs:
        return pd.DataFrame(columns=["path", "emotion", "actor", "dataset"])

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["emotion"])
    return df


def main():
    df = load_all_datasets()
    print("Ukupno audio zapisa učitano:", len(df))
    print(df["dataset"].value_counts())
    print(df.head())


if __name__ == "__main__":
    main()
