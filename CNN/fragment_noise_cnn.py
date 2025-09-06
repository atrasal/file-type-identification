import os
import random
import csv

# ====== CONFIGURATION ======
INPUT_DIR = os.path.join("CNN", "Training_dataset")
OUTPUT_DIR = os.path.join("CNN", "Training_fragments")
CHUNK_SIZE = 1024
REMOVE_HEADER = 64
REMOVE_FOOTER = 64
NOISE_LEVEL = 0.2
NOISE_MODE = "shift"  # Options: xor, delete, shift
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def add_noise(fragment_bytes, noise_level=0.01, mode='xor'):
    array = bytearray(fragment_bytes)
    if mode == 'xor':
        num_noisy = int(len(array) * noise_level)
        for _ in range(num_noisy):
            idx = random.randint(0, len(array) - 1)
            array[idx] ^= random.randint(1, 255)
    elif mode == 'delete':
        num_to_delete = int(len(array) * noise_level)
        for _ in range(num_to_delete):
            idx = random.randint(0, len(array) - 1)
            del array[idx]
        array += bytearray([0] * (CHUNK_SIZE - len(array)))  # pad
    elif mode == 'shift':
        num_shifts = int(len(array) * noise_level)
        for _ in range(num_shifts):
            idx = random.randint(1, len(array) - 1)
            array[idx - 1] = array[idx]
    return bytes(array)

def fragment_file(path, chunk_size, remove_header, remove_footer, noise_level, noise_mode):
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) < (remove_header + remove_footer + chunk_size):
        return []
    data = data[remove_header: -remove_footer]
    fragments = []
    for i in range(0, len(data) - chunk_size + 1, chunk_size):
        chunk = data[i:i + chunk_size]
        if noise_level > 0:
            chunk = add_noise(chunk, noise_level, mode=noise_mode)
        fragments.append(chunk)
    return fragments

def save_hex_fragment(fragment_bytes, path):
    with open(path, 'w') as f:
        hex_string = ''.join(format(b, '02x') for b in fragment_bytes)
        f.write(hex_string)

def process_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping_path = os.path.join(OUTPUT_DIR, "fragment_mapping.csv")
    fragment_counter = 1
    mapping_rows = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, INPUT_DIR)
            file_type = os.path.dirname(rel_path).replace(os.sep, '/')
            fragments = fragment_file(file_path, CHUNK_SIZE, REMOVE_HEADER, REMOVE_FOOTER, NOISE_LEVEL, NOISE_MODE)
            for fragment in fragments:
                out_path = os.path.join(OUTPUT_DIR, f"{fragment_counter}.hex")
                save_hex_fragment(fragment, out_path)
                mapping_rows.append([fragment_counter, file_type])
                fragment_counter += 1
            print(f"{filename}: {len(fragments)} fragments")
    with open(mapping_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fragment_id', 'file_type'])
        writer.writerows(mapping_rows)
    print("✅ Fragmentation complete.")
    print(f"📄 Mapping CSV saved to: {mapping_path}")

if __name__ == "__main__":
    process_dataset()
