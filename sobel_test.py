import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Convert a color image to grayscale if necessary.
    
    Parameters:
    - image: np.ndarray, the input image (grayscale or color).
    
    Returns:
    - grayscale_image: np.ndarray, the grayscale version of the image.
    """
    # Check if the image has 3 channels (color image)
    if len(image.shape) == 3:
        # Convert RGB to grayscale by computing luminance
        grayscale_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        # Image is already grayscale
        grayscale_image = image
    return grayscale_image

def compute_gradient_blocks(grayscale_image: np.ndarray, block_size: int = 4):
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

def visualize_gradient_magnitude_and_angle(image_path: str, block_size: int = 4):
    """
    Reads an image from the given path and visualizes the block-wise gradient magnitude and angle.
    
    Parameters:
    - image_path: str, path to the PNG image file (grayscale or color).
    - block_size: int, size of blocks to divide the image into.
    """
    # Load the image using PIL
    image = Image.open(image_path)
    
    # Convert the image to a numpy array
    image = np.array(image)
    
    # Process the image to grayscale if it's a color image
    grayscale_image = process_image(image)
    
    # Compute block-wise gradient magnitudes and angles
    block_gradients, block_angles = compute_gradient_blocks(grayscale_image, block_size)
    
    # Normalize the gradient magnitude to range [0, 255] for visualization
    block_gradients_normalized = (block_gradients - np.min(block_gradients)) / (np.max(block_gradients) - np.min(block_gradients)) * 255
    
    # Normalize the gradient angle to range [0, 1] for visualization (angle is between -pi and pi)
    block_angles_normalized = (block_angles + np.pi) / (2 * np.pi)  # Shift and normalize to [0, 1]
    
    # Create a figure to display the results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the block-wise gradient magnitude
    ax[0].imshow(block_gradients_normalized, cmap='gray')
    ax[0].set_title(f'Block Gradient Magnitude ({block_size}x{block_size})')
    ax[0].axis('off')  # Turn off axis labels
    
    # Display the block-wise gradient angle using the 'hsv' colormap
    ax[1].imshow(block_angles_normalized, cmap='gray')
    ax[1].set_title(f'Block Gradient Angle ({block_size}x{block_size})')
    ax[1].axis('off')  # Turn off axis labels
    
    # Show the plot
    plt.show()

# Example usage:
image_path = 'validation/frame_0016.png'  # Replace with your image path
visualize_gradient_magnitude_and_angle(image_path, block_size=8)
