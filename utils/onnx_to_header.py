import numpy as np

def write_header(array, name, filename):
    flat_array = array.flatten()
    with open(filename, "w") as f:
        f.write(f"// Auto-generated from {name}.npy\n")
        f.write(f"#ifndef {name.upper()}_H\n#define {name.upper()}_H\n\n")
        f.write(f"#define {name.upper()}_SIZE {flat_array.size}\n\n")
        f.write(f"float {name}[{flat_array.size}] = {{\n")
        for i, val in enumerate(flat_array):
            f.write(f"  {val:.6f},")
            if (i + 1) % 8 == 0:
                f.write("\n")
        f.write("\n};\n\n#endif  // {name.upper()}_H\n")

# Load and convert both input and output arrays
mha_input = np.load("mha_input.npy")
mha_output = np.load("mha_output.npy")

write_header(mha_input, "mha_input", "mha_input.h")
write_header(mha_output, "mha_output", "mha_output.h")

print("Header files generated: mha_input.h, mha_output.h")

#to include in c : #include "mha_input.h"
#include "mha_output.h"

#for (int i = 0; i < MHA_INPUT_SIZE; ++i) {
#    printf("in[%d] = %f, out = %f\n", i, mha_input[i], mha_output[i]);
#}