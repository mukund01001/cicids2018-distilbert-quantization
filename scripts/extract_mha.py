from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from torch.quantization import quantize_dynamic
# Load model and tokenizer
model_name = "chathuru/cicids2018-distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
print(" Model and tokenizer loaded")

# Tokenize input
sample_text = "The source IP triggered multiple alerts on port scan"
inputs = tokenizer(sample_text, return_tensors="pt")
print("Input tokenized")

# Dictionary to hold input/output
mha_io = {}

# Register hook on q_lin layer (gets actual MHA input)
def hook_fn(module, input, output):
    print(" Inside MHA hook!")
    if isinstance(input, tuple) and len(input) > 0:
        mha_io["input"] = input[0].detach()
        print(" Captured input with shape:", mha_io["input"].shape)
    else:
        mha_io["input"] = torch.tensor([])
        print("Could not capture input")
    mha_io["output"] = torch.tensor([])  # Not needed here

print(" Registering hook on model.transformer.layer[0].attention.q_lin...")
mha_layer = model.transformer.layer[0].attention.q_lin
hook = mha_layer.register_forward_hook(hook_fn)

# Run forward pass
print("Running forward pass...")
with torch.no_grad():
    output = model(**inputs)

print(" Forward pass done")
hook.remove()

# Save inputs and output
if "input" in mha_io:
    print("🔸 MHA input shape:", mha_io["input"].shape)
    np.save("mha_input.npy", mha_io["input"].cpu().numpy())
    print("Saved mha_input.npy")

# For MHA output, we register a separate hook on the attention block
def output_hook_fn(module, input, output):
    if isinstance(output, tuple) and len(output) > 0:
        mha_io["output"] = output[0].detach()
    elif isinstance(output, torch.Tensor):
        mha_io["output"] = output.detach()
    else:
        mha_io["output"] = torch.tensor([])

print(" Registering hook to capture MHA output...")
output_hook = model.transformer.layer[0].attention.register_forward_hook(output_hook_fn)

with torch.no_grad():
    output = model(**inputs)

print(" Second forward pass to capture output done")
output_hook.remove()

if "output" in mha_io and mha_io["output"].nelement() > 0:
    print("🔹 MHA output shape:", mha_io["output"].shape)
    np.save("mha_output.npy", mha_io["output"].cpu().numpy())
    print("Saved mha_output.npy")
else:
    print("MHA output not captured")

# Quantize the model (linear layers only)
print(" Quantizing model...")
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), "quantized_distilbert.pt")
print("✅ Quantized model saved as 'quantized_distilbert.pt'")
