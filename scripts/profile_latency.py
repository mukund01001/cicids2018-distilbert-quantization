import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.quantization import quantize_dynamic

# 🧠 Load models
model_fp32 = AutoModelForSequenceClassification.from_pretrained("chathuru/cicids2018-distilbert")
model_int8 = quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.qint8)
model_fp32.eval()
model_int8.eval()

# 📝 Use a single sample from your dataset
sample_text = "0 1234 2345 45.0 30.0 67.8"  # Simulated input (like what you passed from features)
tokenizer = AutoTokenizer.from_pretrained("chathuru/cicids2018-distilbert")
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True)

# ⏱️ FP32 timing
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model_fp32(**inputs)
end = time.time()
fp32_latency = (end - start) / 100

# ⏱️ Quantized timing
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model_int8(**inputs)
end = time.time()
int8_latency = (end - start) / 100

print(f"⚙️ FP32 Avg Latency:    {fp32_latency * 1000:.2f} ms")
print(f"⚙️ Quantized Avg Latency: {int8_latency * 1000:.2f} ms")
