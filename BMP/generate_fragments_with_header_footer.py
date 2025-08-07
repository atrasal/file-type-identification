import os

# Use script's directory as base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train_bmp_images')
TEST_DIR = os.path.join(BASE_DIR, 'test_bmp_images')
OUT_TRAIN = os.path.join(BASE_DIR, 'fragments_with_header_footer', 'train_fragments')
OUT_TEST = os.path.join(BASE_DIR, 'fragments_with_header_footer', 'test_fragments')
FRAGMENT_SIZE = 1024      # 1 KB
STEP_SIZE = 1024          # No overlap
MAX_FRAGMENTS = 10        # At most 10 fragments per image


os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_TEST, exist_ok=True)

def generate_fragments(src_dir, out_dir):
    print(f"Processing directory: {src_dir}")
    if not os.path.exists(src_dir):
        print(f"Directory does not exist: {src_dir}")
        return
    for fname in os.listdir(src_dir):
        print(f"Found file: {fname}")
        if not fname.lower().endswith('.bmp'):
            print(f"Skipping non-BMP file: {fname}")
            continue
        fpath = os.path.join(src_dir, fname)
        print(f"Reading file: {fpath}")
        with open(fpath, 'rb') as f:
            data = f.read()
        file_size = len(data)
        print(f"File size: {file_size} bytes")
        if file_size < FRAGMENT_SIZE:
            print(f"File too small for fragmenting: {fname}")
            continue  # Skip if not enough data
        num_frags = min((file_size - FRAGMENT_SIZE) // STEP_SIZE + 1, MAX_FRAGMENTS)
        print(f"Generating {num_frags} fragments for {fname}")
        for i in range(num_frags):
            start = i * STEP_SIZE
            fragment = data[start:start+FRAGMENT_SIZE]
            if len(fragment) < FRAGMENT_SIZE:
                print(f"Fragment {i+1} too small, skipping.")
                continue
            out_fname = f"{os.path.splitext(fname)[0]}_frag{i+1}.bin"
            out_fpath = os.path.join(out_dir, out_fname)
            with open(out_fpath, 'wb') as out_f:
                out_f.write(fragment)
            print(f"Wrote fragment: {out_fpath}")

if __name__ == '__main__':
    generate_fragments(TRAIN_DIR, OUT_TRAIN)
    generate_fragments(TEST_DIR, OUT_TEST)
