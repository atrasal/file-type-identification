import os

FRAGMENT_SIZE = 1024  # 1 KB
input_dir = 'forensics/BMP/bike/'
output_dir = 'forensics/BMP/bmp_fragments/'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith('.bmp'):
        with open(os.path.join(input_dir, filename), 'rb') as f:
            i = 0
            while True:
                fragment = f.read(FRAGMENT_SIZE)
                if not fragment:
                    break
                frag_name = f"{os.path.splitext(filename)[0]}_frag_{i}.bin"
                with open(os.path.join(output_dir, frag_name), 'wb') as frag_file:
                    frag_file.write(fragment)
                i += 1
print("Fragmentation complete.")
