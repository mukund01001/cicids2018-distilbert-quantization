import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Load model
model = AutoModel.from_pretrained("chathuru/cicids2018-distilbert")
tokenizer = AutoTokenizer.from_pretrained("chathuru/cicids2018-distilbert")

# Set model to eval mode
model.eval()

# Dummy input
text = "This is a test sentence."
inputs = tokenizer(text, return_tensors="pt")

# Hook to capture MHA output
mha_output = {}

def hook_fn(module, input, output):
    mha_output["pytorch"] = output[0].detach().numpy()

handle = model.transformer.layer[5].attention.register_forward_hook(hook_fn)

# Forward pass
with torch.no_grad():
    _ = model(**inputs)

handle.remove()

# Load ONNX MHA output
onnx_mha_output = np.load("mha_output.npy")

# Compare
print("PyTorch shape:", mha_output["pytorch"].shape)
print("ONNX shape:   ", onnx_mha_output.shape)

# Optionally compare values
diff = np.abs(mha_output["pytorch"] - onnx_mha_output)
print("Max diff:", diff.max())
