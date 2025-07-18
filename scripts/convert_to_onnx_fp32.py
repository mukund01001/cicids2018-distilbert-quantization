# inside convert_to_onnx_fp32.py
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class DistilBERTWithMHA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # pick the exact MHA layer
        self.mha = model.transformer.layer[5].attention

    def forward(self, input_ids, attention_mask):
        # Get hidden states
        hidden = self.model.embeddings(input_ids)
        for i in range(6):
            layer = self.model.transformer.layer[i]
            if i == 5:
                mha_output = layer.attention(hidden, attention_mask)[0]
                return mha_output
            else:
                hidden = layer(hidden, attention_mask)[0]

# Load model + wrap
model = AutoModel.from_pretrained("chathuru/cicids2018-distilbert")
wrapper = DistilBERTWithMHA(model)
wrapper.eval()

# Dummy input
tokenizer = AutoTokenizer.from_pretrained("chathuru/cicids2018-distilbert")
inputs = tokenizer("test sample sentence", return_tensors="pt")

# Export only MHA output
torch.onnx.export(
    wrapper,
    (inputs["input_ids"], inputs["attention_mask"]),
    "mha_only.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["mha_output"],
    opset_version=14,
    dynamic_axes={"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"}}
)
