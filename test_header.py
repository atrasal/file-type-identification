import os
import numpy as np

FRAG_DIR = "fragments_dataset"  # Adjust folder if needed
SAMPLES = 500  # Number of random files to inspect

JPEG_HEADER = bytes([0xFF, 0xD8])
JPEG_FOOTER = bytes([0xFF, 0xD9])

def inspect_fragment(path):
    fragment_hex = np.load(path)

    # Convert hex array back to bytes
    try:
        fragment_bytes = bytes([int(h, 16) for h in fragment_hex])
    except Exception as e:
        print(f"[{os.path.basename(path)}] Error decoding hex: {e}")
        return

    has_header = JPEG_HEADER in fragment_bytes[:10]
    has_footer = JPEG_FOOTER in fragment_bytes[-10:]
    metadata_signs = b"JFIF" in fragment_bytes or b"Exif" in fragment_bytes

    print(f"[{os.path.basename(path)}] Header: {has_header} | Footer: {has_footer} | Metadata: {metadata_signs}")

def main():
    files = sorted(f for f in os.listdir(FRAG_DIR) if f.endswith(".npy"))
    for file in files[:SAMPLES]:
        inspect_fragment(os.path.join(FRAG_DIR, file))

if __name__ == "__main__":
    main()
