import random

def add_noise_with_tracking(fragment_bytes, noise_level=0.6):
    array = bytearray(fragment_bytes)
    num_noisy = int(len(array) * noise_level)
    modified_indices = []

    for _ in range(num_noisy):
        idx = random.randint(0, len(array) - 1)
        array[idx] ^= random.randint(1, 255)
        modified_indices.append(idx)

    return bytes(array), modified_indices

def print_diff(original, noisy, modified_indices, max_show=200):
    print(f"\n🧪 Showing first {min(max_show, len(modified_indices))} modified bytes:\n")
    for i in modified_indices[:max_show]:
        print(f"Index {i}: {original[i]:02x} ➡️ {noisy[i]:02x}")

def test_noise_visualization():
    fragment = bytes([random.randint(0, 255) for _ in range(1024)])
    noisy, modified_indices = add_noise_with_tracking(fragment, noise_level=0.6)
    print_diff(fragment, noisy, modified_indices)

if __name__ == "__main__":
    test_noise_visualization()
