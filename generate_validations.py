import numpy as np
from PIL import Image
# Generate a fully black image, run hash function on it

def patch_hash(image: np.ndarray, bin_size: int) -> np.ndarray:
    """
    Privacy hash function where we get the min and max value in every patch
    Assume the input image is already monochrome
    """
    image_hash = np.zeros((256//bin_size, 256//bin_size, 1), dtype=np.uint8)

    # Convert observation to grayscale
    # 64 for 256px image
    # 128 for 512
    reshaped_array = image.reshape(128, 4, 128, 4)

    min_values = reshaped_array.min(axis=(1, 3)) // bin_size
    max_values = reshaped_array.max(axis=(1, 3)) // bin_size

    np.savetxt('min.txt', min_values, fmt='%d')  # '%d' for integers, adjust formatting as needed
    np.savetxt('max.txt', max_values, fmt='%d')  # '%d' for integers, adjust formatting as needed


    for min_val, max_val in zip(min_values.ravel(), max_values.ravel()):
        image_hash[min_val, max_val] = max(image_hash[min_val, max_val] + 1, 255)

    return image_hash


def save_image(image: np.ndarray, name: str) -> np.ndarray:
    image = Image.fromarray(np.squeeze(image), mode="L")
    image_path = f"{name}.png"
    image.save(image_path)


if __name__ == "__main__":
    LENGTH = 512
    BIN_SIZE = 4

    # Fully black image
    black_image = np.zeros((LENGTH, LENGTH), dtype=np.uint8)
    black_image_patch_hash = patch_hash(black_image, BIN_SIZE)

    save_image(black_image, "black_image")
    save_image(black_image_patch_hash, "black_image_patch_hash")

    # Fully white image
    white_image = np.full((LENGTH, LENGTH), 255, dtype=np.uint8)
    white_image_patch_hash = patch_hash(white_image, BIN_SIZE)

    save_image(white_image, "white_image")
    save_image(white_image_patch_hash, "white_image_patch_hash")

    # Half white, half black
    half_image = np.zeros((LENGTH, LENGTH), dtype=np.uint8)
    half_image[:, LENGTH// 2:] = 255
    half_image_patch_hash = patch_hash(half_image, BIN_SIZE)

    save_image(half_image, "half_black")
    save_image(half_image_patch_hash, "half_black_patch_hash")


    # Gradient image
    gradient = np.linspace(0, 255, LENGTH, dtype=np.uint8)
    image = np.tile(gradient, (LENGTH, 1))
    image_patch_hash = patch_hash(image, BIN_SIZE)

    save_image(image, "gradient")
    save_image(image_patch_hash, "gradient_patch_hash")
    

