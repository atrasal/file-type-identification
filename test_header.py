import os
import numpy as np

FRAG_DIR = "fragments_dataset/jpg"  # or any other folder
SAMPLES = 500  # number of random files to inspect

JPEG_HEADER = bytes([0xFF, 0xD8])
JPEG_FOOTER = bytes([0xFF, 0xD9])

def inspect_fragment(path):
    fragment = np.load(path)
    if isinstance(fragment, np.ndarray):
        fragment = fragment.tobytes()

    has_header = JPEG_HEADER in fragment[:10]  # check start
    has_footer = JPEG_FOOTER in fragment[-10:]  # check end
    metadata_signs = b"JFIF" in fragment or b"Exif" in fragment

    print(f"[{os.path.basename(path)}] Header: {has_header} | Footer: {has_footer} | Metadata: {metadata_signs}")

def main():
    files = [f for f in os.listdir(FRAG_DIR) if f.endswith(".npy")]
    samples = files[:SAMPLES]

    for file in samples:
        inspect_fragment(os.path.join(FRAG_DIR, file))

if __name__ == "__main__":
    main()
