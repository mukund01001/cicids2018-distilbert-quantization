import numpy as np

def quantize_array(arr, dtype=np.int8):
    rmin, rmax = arr.min(), arr.max()
    qmin = np.iinfo(dtype).min
    qmax = np.iinfo(dtype).max

    scale = (rmax - rmin) / (qmax - qmin)
    zero_point = np.round(-rmin / scale).astype(int)
    q_arr = np.clip(np.round(arr / scale + zero_point), qmin, qmax).astype(dtype)

    return q_arr, scale, zero_point

def write_header(q_arr, name, filename, dtype="int8_t"):
    flat = q_arr.flatten()
    with open(filename, "w") as f:
        f.write(f"// Quantized {name}.npy\n")
        f.write(f"#ifndef {name.upper()}_H\n#define {name.upper()}_H\n\n")
        f.write(f"#define {name.upper()}_SIZE {flat.size}\n\n")
        f.write(f"{dtype} {name}[{flat.size}] = {{\n")
        for i, val in enumerate(flat):
            f.write(f"  {val},")
            if (i + 1) % 16 == 0:
                f.write("\n")
        f.write("\n};\n\n#endif  // {name.upper()}_H\n")

# Load original .npy files
mha_input = np.load("mha_input.npy")
mha_output = np.load("mha_output.npy")

# Quantize to int8
q_input, input_scale, input_zero = quantize_array(mha_input)
q_output, output_scale, output_zero = quantize_array(mha_output)

# Write headers
write_header(q_input, "mha_input_q", "mha_input_q.h")
write_header(q_output, "mha_output_q", "mha_output_q.h")

# Save scale and zero point for dequantization
with open("quant_params.h", "w") as f:
    f.write(f"#ifndef QUANT_PARAMS_H\n#define QUANT_PARAMS_H\n\n")
    f.write(f"#define INPUT_SCALE {input_scale:.8f}\n")
    f.write(f"#define INPUT_ZERO_POINT {input_zero}\n\n")
    f.write(f"#define OUTPUT_SCALE {output_scale:.8f}\n")
    f.write(f"#define OUTPUT_ZERO_POINT {output_zero}\n\n")
    f.write("#endif // QUANT_PARAMS_H\n")

print("✅ Generated: mha_input_q.h, mha_output_q.h, quant_params.h")
