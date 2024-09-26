import numpy as np
from PIL import Image
import noise
from scipy import ndimage

def patch_hash(image: np.ndarray, bin_size: int, height: int, width: int, patch_size: int) -> np.ndarray:
    """
    Privacy hash function where we get the min and max value in every patch
    Assume the input image is already monochrome
    """
    image_hash = np.zeros((256//bin_size, 256//bin_size, 1), dtype=np.uint16)

    # Reshape so its faster to calculate the min and max in each patch
    reshaped_array = image.reshape(height//patch_size, patch_size, width//patch_size, patch_size)


    min_values = reshaped_array.min(axis=(1, 3)) // bin_size
    max_values = reshaped_array.max(axis=(1, 3)) // bin_size

    for min_val, max_val in zip(min_values.ravel(), max_values.ravel()):
        image_hash[min_val, max_val] = min(image_hash[min_val, max_val] + 1, 255)

    return image_hash

def gradient_blocks_hash(grayscale_image: np.ndarray, block_size: int = 8):
    """
    Compute the gradient magnitude and angle in each block of the image.
    
    Parameters:
    - grayscale_image: np.ndarray, the grayscale version of the image.
    - block_size: int, the size of the blocks (default is 4x4).
    
    Returns:
    - block_gradients: np.ndarray, the summarized gradient magnitudes for each block.
    - block_angles: np.ndarray, the summarized gradient angles for each block.
    """
    
    # Compute the gradient in the x and y directions using Sobel operator
    sobel_x = ndimage.sobel(grayscale_image, axis=0)  # Gradient in the x-direction
    sobel_y = ndimage.sobel(grayscale_image, axis=1)  # Gradient in the y-direction
    
    # Compute the gradient magnitude and angle
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_angle = np.arctan2(sobel_y, sobel_x)  # Gradient angle in radians
    
    # Initialize lists to store block-level gradient summaries
    block_gradients = []
    block_angles = []
    
    # Split the image into blocks and compute the summary of each block
    for i in range(0, grayscale_image.shape[0], block_size):
        for j in range(0, grayscale_image.shape[1], block_size):
            # Extract the block for magnitude and angle
            block_magnitude = gradient_magnitude[i:i+block_size, j:j+block_size]
            block_angle = gradient_angle[i:i+block_size, j:j+block_size]
            
            # Compute the summary (e.g., mean) for the block
            block_grad_summary = np.mean(block_magnitude)  # You can use np.max or another statistic
            block_angle_summary = np.mean(block_angle)     # Mean angle for the block
            
            # Store the summarized values for each block
            block_gradients.append(block_grad_summary)
            block_angles.append(block_angle_summary)
    
    # Convert the lists to numpy arrays (reshape to match image layout)
    height, width = grayscale_image.shape
    block_gradients = np.array(block_gradients).reshape(height // block_size, width // block_size)
    block_angles = np.array(block_angles).reshape(height // block_size, width // block_size)
    
    return block_gradients, block_angles


def save_image(image: np.ndarray, name: str, upscale_factor: int = 1, gamma: float = 0.6, normalise: bool = True) -> None:
    """
    Save an image after normalizing it to the range 0-255.
    
    Optionally upscale the image by a given factor.
    
    Parameters:
    - image: np.ndarray, the input image array (can be 2D or 3D).
    - name: str, the name of the saved image file.
    - upscale_factor: int, the factor by which to upscale the image (default is 1, no upscaling).
    - gamma: float, the gamma correction value (default is 0.6).
    - normalise: bool, whether to normalize the image (default is True).
    """
    # If the image is 3D (multiple layers), normalize each layer separately and hstack the layers
    if len(image.shape) == 3:
        layers = []
        for layer in range(image.shape[2]):
            single_layer = image[:, :, layer]
            if np.max(single_layer) != 0 and normalise:
                single_layer = (single_layer - np.min(single_layer)) / (np.max(single_layer) - np.min(single_layer)) * 255
            single_layer = apply_gamma_correction(single_layer.astype(np.uint8), gamma)
            layers.append(single_layer)
        
        # Stack layers horizontally (hstack) to save as a single image
        image = np.hstack(layers)
    
    # If the image is 2D, just normalize and apply gamma correction
    else:
        if np.max(image) != 0 and normalise:  # Avoid division by zero if the max value is 0
            image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = apply_gamma_correction(image.astype(np.uint8), gamma)

    # Convert the normalized array to a PIL image
    image = Image.fromarray(np.squeeze(image), mode="L")


    # Upscale the image if upscale_factor is greater than 1
    if upscale_factor > 1:
        width, height = image.size
        new_size = (width * upscale_factor, height * upscale_factor)
        image = image.resize(new_size, Image.NEAREST)  # Use nearest neighbor interpolation for upscaling

    # Save the image
    image_path = f"{name}.png"
    image.save(image_path)

def apply_gamma_correction(image: np.ndarray, gamma: int):
    """
    Apply gamma correction to brighten or darken the image, < 1 makes dark regions lighter
    """
    # Check if the gamma value is valid (it should be positive)
    if gamma <= 0:
        raise ValueError("Gamma value must be greater than 0.")

    # Step 1: Normalize the image to the range [0, 1]
    normalized_image = image / 255.0
    
    # Step 2: Apply gamma correction
    corrected_image = np.power(normalized_image, gamma)
    
    # Step 3: Rescale back to [0, 255] and convert to unsigned 8-bit integer
    corrected_image = np.uint8(corrected_image * 255)
    
    return corrected_image


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

def load_rgb_image(filepath) -> np.ndarray:
    """Load an RGB image and convert it to a grayscale NumPy array (0-255)."""
    # Load the image
    image = Image.open(filepath)
    
    # Convert the image to grayscale
    grayscale_image = image.convert("L")  # "L" mode is for grayscale
    
    # Convert the grayscale image to a NumPy array
    grayscale_array = np.array(grayscale_image, dtype=np.uint8)
    
    return grayscale_array

def process_and_save_image(image: np.ndarray, image_name, bin_size):
    """Generate patch hash and save the image and its hash."""
    image_patch_hash = patch_hash(image, bin_size, patch_size=4, height=image.shape[0], width=image.shape[1])
    gradient, angle  = gradient_blocks_hash(image, block_size=8)
    combined_gradient = np.stack((gradient, angle), axis=-1)
    # combined_gradient = np.stack((gradient), axis=-1)
    # np.savetxt(f"{image_name}.txt", np.squeeze(image_patch_hash), fmt="%d")
    # save_image(image, image_name)
    # save_image(image_patch_hash, f"{image_name}_patch_hash", upscale_factor=8)
    # save_image(angle, f"{image_name}_angle_hash", upscale_factor=8, gamma=1)
    save_image(gradient, f"{image_name}_gradient_hash", upscale_factor=8, gamma=1)
    save_image(combined_gradient, f"{image_name}_combined_gradient_angle_hash", upscale_factor=8, gamma=1)

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
    HEIGHT = 512
    WIDTH = 512
    BIN_SIZE = 2

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

    # Load a camera frame
    camera_image = load_rgb_image("frame_0016.png")
    process_and_save_image(camera_image, "frame_0016_", BIN_SIZE)

    # Load a camera frame
    camera_image = load_rgb_image("frame_0015.png")
    process_and_save_image(camera_image, "frame_0015_", BIN_SIZE)

    # IRL image of a home
    camera_image = load_rgb_image("home.jpg")
    process_and_save_image(camera_image, "home_", BIN_SIZE)
    

