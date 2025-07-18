# 🚨 Transformer-Based Network Intrusion Detection on Edge

## 🧠 Overview

This project implements a Transformer-based Network Intrusion Detection System (NIDS), specifically optimized for edge deployment. It serves as the official software stack supporting our custom hardware accelerator designed for the Multi-Head Attention (MHA) layer of DistilBERT. This work was developed for the **DVCon India 2025 Design Contest**, under team **I2CS\_tvaritas**, and we are proud to be **Finalists**, having completed the **synthesis stage** and currently proceeding toward final integration.

The model has been trained and evaluated on the CICIDS2018 dataset, a widely used benchmark for network attack detection. The system demonstrates significant performance improvements through quantization, layer-wise optimization, and hardware-software co-design, with the ultimate goal of deploying a real-time, low-latency IDS on RISC-V SoCs like the Vega AT105.

## 🎯 Project Motivation

Transformer models offer excellent accuracy for intrusion detection, but their compute-heavy structure traditionally makes them unsuitable for real-time inference on resource-constrained edge devices. Our primary objectives were to:

- Quantize a pretrained DistilBERT model to int8 for significantly reduced memory footprint and faster inference.
- Extract and profile the Multi-Head Attention (MHA) layer to enable seamless interfacing with our custom hardware accelerator IP.
- Deploy and simulate the full pipeline—from raw dataset ingestion to inference and accelerator-ready input/output extraction.
- Enable real-time anomaly detection tailored for IoT and embedded systems using RISC-V platforms.

## 🧪 Dataset: CICIDS2018

- **Source**: Canadian Institute for Cybersecurity
- **Format**: Raw `.csv` files; preprocessed into text-label format for model input.
- **Hosted**: [📁 Download Data Folder](https://drive.google.com/drive/folders/1HnCT0Jbx0cCsxXtkuJFOY0rd0a4j9Bd4?usp=sharing)

## 🏗️ Repository Structure

```
.
├── data/                     # Raw and processed CICIDS2018 CSVs
├── models/                   # Quantized and ONNX DistilBERT models (external)
├── scripts/                  # Evaluation, quantization, profiling, and data prep scripts
├── utils/                    # ONNX ↔ NumPy conversion and verification utilities
├── headers/                  # Output tensor dumps in .h and .npy format
├── reports/                  # Evaluation reports in .md and .docx
├── requirements.txt          # Python dependencies
└── README.md
```

## ⚙️ Key Components

### ✅ Model Quantization

Performed using Post-Training Quantization (PTQ) via ONNX and PyTorch:

- `scripts/quantize_model.py`: PTQ implementation
- `scripts/convert_to_onnx_fp32.py`: Convert FP32 PyTorch → ONNX
- `scripts/eval_fp32.py` / `eval_quantized.py`: Accuracy & latency evaluation

### 🧩 MHA Layer Extraction

- `scripts/extract_mha.py`: Captures exact inputs/outputs of DistilBERT’s MHA layer
- Dumps to `.h` and `.npy` formats for hardware simulation and interfacing

### 🔁 ONNX ↔ NumPy Interoperability

- `utils/onnx_to_header.py`: Converts ONNX tensors to C headers
- `utils/compare onnx to npy.py`: Ensures ONNX and NumPy consistency

### 📉 Reports & Profiling

- `scripts/profile_latency.py`: Profiles inference time
- `reports/model_eval_report.md`: Detailed accuracy and latency benchmarks

### 🔨 Hardware Accelerator

Our custom accelerator targets the MHA layer with these specs:

- Written in C++ using Vitis HLS for FPGA
- Implements tiled QKᵀ and QKVV matrix ops
- AXI-Lite + AXI4 interfaces for SoC integration
- Synthesizable for the Vega AT105 RISC-V SoC

📄 **Design Reports**:

- [`Transformer Based Network Intrusion Detection.pdf`](docs/Transformer%20Based%20Network%20Intrusion%20Detection.pdf)
- [`DVCon2025 Stage 2 Report.pdf`](docs/DVCon2025%20Stage%202%20Report.pdf)

## 📊 Results

| Model Version | Accuracy | Avg Inference Time |
| ------------- | -------- | ------------------ |
| FP32          | 91.00%   | \~550 ms           |
| INT8 PTQ      | 96.00%   | \~310 ms           |

## 📦 Installation

```bash
# Clone repository
$ git clone https://github.com/mukund01001/cicids2018-distilbert-quantization.git
$ cd cicids2018-distilbert-quantization

# Create virtual environment (optional)
$ python -m venv venv
$ source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
$ pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Preprocess CSV data
python scripts/prepare_data.py

# Evaluate FP32 model
python scripts/eval_fp32.py

# Quantize and evaluate
python scripts/quantize_model.py
python scripts/eval_quantized.py

# Extract MHA tensors
python scripts/extract_mha.py

# Convert ONNX → Header
python utils/onnx_to_header.py

# Profile inference latency
python scripts/profile_latency.py
```

> 🗂️ Make sure to manually download large files:
>
> - [📁 Models (Google Drive)](https://drive.google.com/drive/folders/1pZDrxV7i3kbrZ3b4L3tWVGBBtgp2zeZ0?usp=sharing)
> - [📁 Data (Google Drive)](https://drive.google.com/drive/folders/1HnCT0Jbx0cCsxXtkuJFOY0rd0a4j9Bd4?usp=sharing)

## 📚 References

- [CICIDS2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)
- [HuggingFace model – chathuru/cicids2018-distilbert](https://huggingface.co/chathuru/cicids2018-distilbert)
- Li et al., "FPGA Accelerator for Transformer" – arXiv:2403.16731
- Yang X, Su T. "EFA-Trans", Electronics 2022
- Xilinx UG1399: Vitis HLS

## 👨‍💻 Authors

Team **I2CS\_tvaritas**, IIIT Kottayam\
**Mukund Rathi**\
**Joel Dan Philip**\
**Shravan Narayan Sunil**

For questions or collaboration:  [rathimukund.01@gmail.com](mailto\:rathimukund.01@gmail.com)

---

⭐ If you found this project useful, please consider starring the repo!

