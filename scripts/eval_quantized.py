import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.quantization import quantize_dynamic
import pandas as pd
from sklearn.metrics import accuracy_score

# 🔧 Load and quantize the model
base_model = AutoModelForSequenceClassification.from_pretrained("chathuru/cicids2018-distilbert")
quantized_model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
quantized_model.eval()

tokenizer = AutoTokenizer.from_pretrained("chathuru/cicids2018-distilbert")

# 📥 Load data
df = pd.read_csv("cicids2018_test.csv")
df = df.head(500)

texts = df["text"].tolist()
labels = df["label"].tolist()

# 🔁 Inference loop
preds = []
for idx, text in enumerate(texts):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = quantized_model(**inputs)
        pred = torch.argmax(output.logits, dim=1).item()
        preds.append(pred)

    if idx % 100 == 0:
        print(f"Processed {idx}/{len(texts)}")

# ✅ Accuracy
acc = accuracy_score(labels, preds)
print(f"✅ PTQ Quantized Accuracy: {acc:.4f}")
