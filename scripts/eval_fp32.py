import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score

model = AutoModelForSequenceClassification.from_pretrained("chathuru/cicids2018-distilbert")
tokenizer = AutoTokenizer.from_pretrained("chathuru/cicids2018-distilbert")
model.eval()

df = pd.read_csv("cicids2018_test.csv")

# 🔁 Reduce sample size temporarily if slow
df = df.head(500)  # Use full dataset later

texts = df["text"].tolist()
labels = df["label"].tolist()

preds = []
for idx, text in enumerate(texts):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
        pred = torch.argmax(output.logits, dim=1).item()
        preds.append(pred)

    # 📢 Show progress
    if idx % 100 == 0:
        print(f"Processed {idx}/{len(texts)}")

acc = accuracy_score(labels, preds)
print(f"✅ FP32 Accuracy: {acc:.4f}")
