# prepare_data.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1️⃣ Adjust this path to where you’ve downloaded CICIDS2018 CSVs
DATA_DIR = r"C:\Users\maila\OneDrive\Desktop\quantize"

# 2️⃣ List the CSV files you want to include in your test set
CSV_FILES = [
    "02-14-2018.csv",
    # add more if desired
]

# 3️⃣ The columns your DistilBERT model was fine‑tuned on
FEATURE_COLUMNS = [
    "Flow Duration", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd IAT Mean", "Bwd IAT Mean", "Fwd Pkt Len Mean",
    # … add all features used during fine‑tuning
]

LABEL_COLUMN = "Label"

def load_and_prepare():
    dfs = []
    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)

    # Convert binary label (BENIGN vs Attack)
    df = df[[*FEATURE_COLUMNS, LABEL_COLUMN]].dropna()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(lambda x: 0 if x == "BENIGN" else 1)

    # Build text inputs
    texts = df[FEATURE_COLUMNS] \
        .astype(str) \
        .agg(" ".join, axis=1) \
        .tolist()

    labels = df[LABEL_COLUMN].tolist()
    return texts, labels

if __name__ == "__main__":
    texts, labels = load_and_prepare()
    print(f"Loaded {len(texts)} samples")
    # Optionally split into train/test if you need
    # X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    # And save to disk:
    pd.DataFrame({"text": texts, "label": labels}) \
      .to_csv("cicids2018_test.csv", index=False)
    print("Saved cicids2018_test.csv")
