import torch
from transformers import AutoModel
from torch.quantization import quantize_dynamic

print("Starting quantization...")

# Load model
model = AutoModel.from_pretrained("chathuru/cicids2018-distilbert")
model.eval()
print(" Model loaded")

# Quantize (only Linear layers)
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
print("Quantization applied")

# Save quantized model
torch.save(quantized_model.state_dict(), "quantized_distilbert.pt")
print("Saved quantized model as 'quantized_distilbert.pt'")
