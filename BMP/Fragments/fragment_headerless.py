import os

HEADER_SKIP = 100
FRAGMENT_SIZE = 1024
input_dir = 'forensics/BMP/bike/'
output_dir = 'forensics/BMP/bmp_headerless/'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith('.bmp'):
        with open(os.path.join(input_dir, filename), 'rb') as f:
            f.seek(HEADER_SKIP)
            fragment = f.read(FRAGMENT_SIZE)
            if fragment:
                frag_name = f"{os.path.splitext(filename)[0]}_headerless.bin"
                with open(os.path.join(output_dir, frag_name), 'wb') as frag_file:
                    frag_file.write(fragment)
print("Headerless fragmentation complete.")
