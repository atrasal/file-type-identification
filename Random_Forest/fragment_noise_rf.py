import os
import random
import csv

# ====== CONFIGURATION ======
INPUT_DIR = os.path.join("Random_Forest", "Training_dataset")
OUTPUT_DIR = os.path.join("Random_Forest", "Training_fragments")
CHUNK_SIZE = 1024
STEP_SIZE = 256   # overlap step (smaller = more fragments)
REMOVE_HEADER = 64
REMOVE_FOOTER = 64
NOISE_LEVEL = 0.2
NOISE_MODE = "shift"  # Options: xor, delete, shift
RANDOM_SEED = 42
FRAGMENTS_PER_TYPE = 5000
AUDIO_EXTENSIONS = {"aac", "flac", "mp3", "ogg", "opus", "wav", "wma", "amr"}

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


def fragment_file(path, chunk_size, step_size, remove_header, remove_footer, noise_level, noise_mode):
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) < chunk_size:
        return []
    data = data[remove_header: len(data) - remove_footer if remove_footer else None]
    fragments = []
    for i in range(0, len(data) - chunk_size + 1, step_size):
        chunk = data[i:i + chunk_size]
        if noise_level > 0:
            chunk = add_noise(chunk, noise_level, mode=noise_mode)
        fragments.append(chunk)
    return fragments


def save_hex_fragment(fragment_bytes, path):
    with open(path, 'w') as f:
        hex_string = ''.join(format(b, '02x') for b in fragment_bytes)
        f.write(hex_string)


def normalize_extension(ext):
    """Normalize extensions to merge only tif/tiff and audios."""
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    elif ext in {"tif", "tiff"}:
        return "tiff"
    else:
        return ext   # keep jpg and jpeg separate


def process_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping_path = os.path.join(OUTPUT_DIR, "fragment_mapping.csv")
    fragment_counter = 1
    mapping_rows = []
    fragments_by_type = {}

    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, INPUT_DIR)
            ext = os.path.splitext(filename)[1][1:].lower()
            if not ext:
                continue

            file_type = normalize_extension(ext)
            fragments = fragment_file(
                file_path, CHUNK_SIZE, STEP_SIZE,
                REMOVE_HEADER, REMOVE_FOOTER,
                NOISE_LEVEL, NOISE_MODE
            )

            if file_type not in fragments_by_type:
                fragments_by_type[file_type] = []
            fragments_by_type[file_type].extend(fragments)

            print(f"{rel_path}: {len(fragments)} fragments")

    # save balanced fragments
    for file_type, fragments in fragments_by_type.items():
        if len(fragments) > FRAGMENTS_PER_TYPE:
            selected = random.sample(fragments, FRAGMENTS_PER_TYPE)
        else:
            selected = fragments
        for fragment in selected:
            out_path = os.path.join(OUTPUT_DIR, f"{fragment_counter}.hex")
            save_hex_fragment(fragment, out_path)
            mapping_rows.append([fragment_counter, file_type])
            fragment_counter += 1
        print(f"Saved {len(selected)} fragments for type: {file_type}")

    with open(mapping_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fragment_id', 'file_type'])
        writer.writerows(mapping_rows)

    print("✅ Training Fragmentation complete.")
    print(f"📄 Mapping CSV saved to: {mapping_path}")


if __name__ == "__main__":
    process_dataset()
