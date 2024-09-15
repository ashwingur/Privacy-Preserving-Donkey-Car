import numpy as np
from PIL import Image
import noise

def patch_hash(image: np.ndarray, bin_size: int, height: int, width: int, patch_size: int) -> np.ndarray:
    """
    Privacy hash function where we get the min and max value in every patch
    Assume the input image is already monochrome
    """
    image_hash = np.zeros((256//bin_size, 256//bin_size, 1), dtype=np.uint8)

    # Convert observation to grayscale
    # 64 for 256px image
    # 128 for 512
    reshaped_array = image.reshape(height//patch_size, patch_size, width//patch_size, patch_size)

    min_values = reshaped_array.min(axis=(1, 3)) // bin_size
    max_values = reshaped_array.max(axis=(1, 3)) // bin_size

    for min_val, max_val in zip(min_values.ravel(), max_values.ravel()):
        image_hash[min_val, max_val] = min(image_hash[min_val, max_val] + 1, 255)

    return image_hash


def save_image(image: np.ndarray, name: str) -> None:
    # Normalize the image to the range 0-255 if it's not already in that range
    if np.max(image) != 0:  # Avoid division by zero if the max value is 0
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = image.astype(np.uint8)  # Convert to uint8 after normalization

    # Convert the normalized array to an image and save it
    image = Image.fromarray(np.squeeze(image), mode="L")
    image_path = f"{name}.png"
    image.save(image_path)


def create_black_image(height):
    """Create a fully black image."""
    return np.zeros((height, height), dtype=np.uint8)

def create_white_image(height):
    """Create a fully white image."""
    return np.full((height, height), 255, dtype=np.uint8)

def create_half_black_half_white_image(height):
    """Create an image where the left half is black and the right half is white."""
    image = np.zeros((height, height), dtype=np.uint8)
    image[:, height // 2:] = 255
    return image

def create_gradient_image(height):
    """Create an image with a horizontal gradient from 0 to 255."""
    gradient = np.linspace(0, 255, height, dtype=np.uint8)
    return np.tile(gradient, (height, 1))

def process_and_save_image(image: np.ndarray, image_name, bin_size):
    """Generate patch hash and save the image and its hash."""
    image_patch_hash = patch_hash(image, bin_size, patch_size=4, height=image.shape[0], width=image.shape[1])
    np.savetxt(f"{image_name}.txt", np.squeeze(image_patch_hash), fmt="%d")
    # print(image_patch_hash)
    save_image(image, image_name)
    save_image(image_patch_hash, f"{image_name}_patch_hash")

def create_random_noise_image(height, width, noise_range=(0, 255)):
    """
    Create an image with random noise.
    
    Parameters:
    - height: int, the height of the image.
    - width: int, the width of the image.
    - noise_range: tuple, the range of noise values (default is (0, 255) for grayscale).
    
    Returns:
    - np.ndarray: A 2D array representing the noise image.
    """
    return np.random.randint(noise_range[0], noise_range[1], (height, width), dtype=np.uint8)

def create_perlin_noise_image(height, width, scale=100, octaves=6, persistence=0.5, lacunarity=2.0):
    """
    Create an image with Perlin noise.
    
    Parameters:
    - height: int, the height of the image.
    - width: int, the width of the image.
    - scale: float, the scale of the noise (larger scale gives smoother noise).
    - octaves: int, number of levels of detail.
    - persistence: float, amplitude multiplier for each octave.
    - lacunarity: float, frequency multiplier for each octave.
    
    Returns:
    - np.ndarray: A 2D array representing the Perlin noise image.
    """
    # Create an empty array for the noise
    perlin_noise = np.zeros((height, width), dtype=np.float32)

    # Generate Perlin noise values for each pixel
    for i in range(height):
        for j in range(width):
            x = i / scale
            y = j / scale
            perlin_noise[i][j] = noise.pnoise2(x, y, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=0)
    
    # Normalize the noise to fit in the range [0, 255]
    perlin_noise_normalized = np.interp(perlin_noise, (perlin_noise.min(), perlin_noise.max()), (0, 255)).astype(np.uint8)

    return perlin_noise_normalized

if __name__ == "__main__":
    HEIGHT = 256
    WIDTH = 256
    BIN_SIZE = 4

    # Fully black image
    black_image = create_black_image(HEIGHT)
    process_and_save_image(black_image, "black_image", BIN_SIZE)

    # Fully white image
    white_image = create_white_image(HEIGHT)
    process_and_save_image(white_image, "white_image", BIN_SIZE)

    # Half white, half black image
    half_image = create_half_black_half_white_image(HEIGHT)
    process_and_save_image(half_image, "half_black", BIN_SIZE)

    # Gradient image
    gradient_image = create_gradient_image(HEIGHT)
    process_and_save_image(gradient_image, "gradient", BIN_SIZE)

    # Create and save random noise image
    noise_image = create_random_noise_image(HEIGHT, WIDTH)
    process_and_save_image(noise_image, "random_noise", BIN_SIZE)

    # Create and save Perlin noise image
    perlin_noise_image = create_perlin_noise_image(HEIGHT, WIDTH)
    process_and_save_image(perlin_noise_image, "perlin_noise", BIN_SIZE)
    

