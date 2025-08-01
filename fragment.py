import os
import numpy as np
import random

# ====== CONFIGURATION ======
INPUT_DIR = "images_dataset"   # Root folder with jpg/, png/, etc.
OUTPUT_DIR = "fragments_dataset"
CHUNK_SIZE = 1024              # Size of each fragment in bytes
REMOVE_HEADER = 64             # Bytes to remove from start of file
REMOVE_FOOTER = 64             # Bytes to remove from end of file
NOISE_LEVEL = 0.01             # Percentage of bytes to corrupt (0.01 = 1%)
RANDOM_SEED = 42               # For reproducibility

random.seed(RANDOM_SEED)

def add_noise(fragment_bytes, noise_level=0.01):
    array = np.frombuffer(fragment_bytes, dtype=np.uint8).copy()  # Make writable
    num_noisy = int(len(array) * noise_level)
    for _ in range(num_noisy):
        idx = random.randint(0, len(array) - 1)
        array[idx] ^= random.randint(1, 255)  # Random XOR noise
    return array.tobytes()


# ====== FRAGMENT A FILE INTO CHUNKS ======
def fragment_file(path, chunk_size, remove_header, remove_footer, noise_level):
    with open(path, 'rb') as f:
        data = f.read()

    # Strip header and footer
    if len(data) < (remove_header + remove_footer + chunk_size):
        return []  # Skip too small files

    data = data[remove_header: -remove_footer]
    
    fragments = []
    for i in range(0, len(data) - chunk_size + 1, chunk_size):
        chunk = data[i:i + chunk_size]
        if noise_level > 0:
            chunk = add_noise(chunk, noise_level)
        vector = np.frombuffer(chunk, dtype=np.uint8)
        fragments.append(vector[:chunk_size])
    return fragments

# ====== PROCESS ALL FILES IN FOLDER STRUCTURE ======
def process_dataset(input_dir, output_dir, chunk_size, remove_header, remove_footer, noise_level):
    os.makedirs(output_dir, exist_ok=True)

    for filetype in os.listdir(input_dir):
        type_path = os.path.join(input_dir, filetype)
        if not os.path.isdir(type_path):
            continue

        out_type_path = os.path.join(output_dir, filetype)
        os.makedirs(out_type_path, exist_ok=True)

        for filename in os.listdir(type_path):
            file_path = os.path.join(type_path, filename)
            fragments = fragment_file(file_path, chunk_size, remove_header, remove_footer, noise_level)

            for idx, fragment in enumerate(fragments):
                out_path = os.path.join(out_type_path, f"{filename}_frag{idx}.npy")
                np.save(out_path, fragment)

            print(f"{filename}: {len(fragments)} fragments")

    print("✅ Fragmentation complete.")

# ====== RUN ======
if __name__ == "__main__":
    process_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        chunk_size=CHUNK_SIZE,
        remove_header=REMOVE_HEADER,
        remove_footer=REMOVE_FOOTER,
        noise_level=NOISE_LEVEL
    )
