"""
file: donkey_env.py
author: Tawn Kramer
date: 2018-08-31
"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import random
import os
import cv2
from PIL import Image
import imageio.v2 as imageio
import shutil

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


def supply_defaults(conf: Dict[str, Any]) -> None:
    """
    Update the config dictonnary
    with defaults when values are missing.

    :param conf: The user defined config dict,
        passed to the environment constructor.
    """
    defaults = [
        ("start_delay", 5.0),
        ("max_cte", 8.0),
        ("frame_skip", 1),
        ("cam_resolution", (120, 160, 3)),
        ("log_level", logging.INFO),
        ("host", "localhost"),
        ("port", 9091),
        ("steer_limit", 1.0),
        ("throttle_min", 0.0),
        ("throttle_max", 1.0),
        ("privacy", False),
        ("record", False),
    ]

    for key, val in defaults:
        if key not in conf:
            conf[key] = val
            print(f"Setting default: {key} {val}")


class DonkeyEnv(gym.Env):
    """
    OpenAI Gym Environment for Donkey

    :param level: name of the level to load
    :param conf: configuration dictionary
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    ACTION_NAMES: List[str] = ["steer", "throttle"]
    VAL_PER_PIXEL: int = 255

    def __init__(self, level: str, conf: Optional[Dict[str, Any]] = None):
        print("starting DonkeyGym env")
        self.viewer = None
        self.proc = None

        if conf is None:
            conf = {}

        conf["level"] = level

        # ensure defaults are supplied if missing.
        supply_defaults(conf)

        # set logging level
        logging.basicConfig(level=conf["log_level"])

        # Bin size for privacy histogram
        self.bin_size = 4

        logger.debug("DEBUG ON")
        logger.debug(conf)

        self.is_privacy = conf["privacy"]
        self.is_record = conf["record"]
        print(f"Donkey privacy set: {self.is_privacy}")
        print(f"Recording enabled: {self.is_record}")

        if self.is_record:
            image_folders = ['frames/', 'privacy_frames/']
            for image_folder in image_folders:
                # Clear the directory if it exists
                if os.path.exists(image_folder):
                    shutil.rmtree(image_folder)
                os.makedirs(image_folder, exist_ok=True)

        # start Unity simulation subprocess
        self.proc = None
        if "exe_path" in conf:
            self.proc = DonkeyUnityProcess()
            # the unity sim server will bind to the host ip given
            self.proc.start(conf["exe_path"], host="0.0.0.0", port=conf["port"])

            # wait for simulator to startup and begin listening
            time.sleep(conf["start_delay"])

        # start simulation com
        self.viewer = DonkeyUnitySimContoller(conf=conf)

        # Note: for some RL algorithms, it would be better to normalize the action space to [-1, 1]
        # and then rescale to proper limtis
        # steering and throttle
        self.action_space = spaces.Box(
            low=np.array([-float(conf["steer_limit"]), float(conf["throttle_min"])]),
            high=np.array([float(conf["steer_limit"]), float(conf["throttle_max"])]),
            dtype=np.float32,
        )

        # camera sensor data
        if self.is_privacy:
            self.observation_space = self.get_privacy_observation_space()
        else:
            self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = conf["frame_skip"]

        # wait until the car is loaded in the scene
        self.viewer.wait_until_loaded()

        self.frame_number = 0
        self.images_array = []
        self.privacy_images_array = []
        self.x_positions = []
        self.z_positions = []

    def plot_xy(self, fig_name: str):
        """
        Creates a scatter plot of the X,Y coordinates of a training/testing run and saves it to a file.
        """
        plt.scatter(self.x_positions, self.z_positions, color='blue', marker='o', s=5)

        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')

        plt.savefig(fig_name, format='png')
        print(f"XY data plotted and saved to {fig_name}")

        # Save as EPS
        plt.savefig(f"{fig_name}.eps", format='eps')
        print(f"XY data plotted and saved to {fig_name}.eps")

        plt.close()
    
    def poop(self):
        print("POOP")

    def save_frame(self, observation, id: int):
        # Convert the observation (which is a NumPy array) to an image
        image = Image.fromarray(observation)
        image_path = f"frames/frame_{id:04d}.png"
        image.save(image_path)
        self.images_array.append(image_path)

    def apply_gamma_correction(self, image: np.ndarray, gamma: int):
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

    def save_privacy_frame(self, image: np.ndarray, id: int):
        # Convert the observation (which is a NumPy array) to an image
        # Normalize the image to the range 0-255 if it's not already in that range
        gamma = 1
        normalise = True
        if len(image.shape) == 3:
            layers = []
            for layer in range(image.shape[2]):
                single_layer = image[:, :, layer]
                if np.max(single_layer) != 0 and normalise:
                    single_layer = (single_layer - np.min(single_layer)) / (np.max(single_layer) - np.min(single_layer)) * 255
                single_layer = self.apply_gamma_correction(single_layer.astype(np.uint8), gamma)
                layers.append(single_layer)
            # Stack layers horizontally (hstack) to save as a single image
            image = np.hstack(layers)
        # If the image is 2D, just normalize
        else:
            if np.max(image) != 0 and normalise:  # Avoid division by zero if the max value is 0
                image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            image = self.apply_gamma_correction(image.astype(np.uint8), gamma)
        image = Image.fromarray(np.squeeze(image*1), mode="L")
        image_path = f"privacy_frames/frame_{id:04d}.png"
        image.save(image_path)
        self.privacy_images_array.append(image_path)

    
    def generate_mp4_video(images_array, video_name='output_video.mp4', fps=30, scale_factor=2):
        """
        Create an MP4 video from a list of image file paths, upscaling each image using nearest-neighbor interpolation.

        :param images_array: List of file paths to images.
        :param video_name: Name of the output video file.
        :param fps: Frames per second.
        :param scale_factor: The factor by which to upscale the images.
        """
        if len(images_array) == 0:
            print("No images provided")
            return

        # Read the first image to get the dimensions
        first_image = imageio.imread(images_array[0])
        height, width, layers = first_image.shape
        new_height, new_width = height * scale_factor, width * scale_factor

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))
        for image_file in images_array:
            image = imageio.imread(image_file)
            # Upscale using nearest-neighbor interpolation
            upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            video.write(cv2.cvtColor(upscaled_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

        video.release()
        print(f"Video saved as {video_name}")


    # def get_privacy_observation_space(self) -> spaces.Box:
    #     return spaces.Box(0, 255, (256//self.bin_size, 256//self.bin_size, 1), dtype=np.uint8)

    def get_privacy_observation_space(self) -> spaces.Box:
        # Changed based on the output of the hash function
        return spaces.Box(0, 255, (64, 64, 2), dtype=np.uint8)
    
    def observation_to_privacy_observation(self, observation: np.ndarray) -> np.ndarray:
        gray_image = np.dot(observation[...,:3], [0.2989, 0.5870, 0.1140])
        gray_image = np.expand_dims(gray_image.astype(np.uint8), axis=-1)

        return self.line_min_max_hash(gray_image, segment_size=16, bin_size=self.bin_size)

        # Gradient hash
        
        # return self.gradient_blocks_hash(gray_image, block_size=8)
    
    def line_min_max_hash(self, image: np.ndarray, segment_size: int, bin_size: int) -> np.ndarray:
        """
        Optimized function to find the min and max of horizontal and vertical segments of a given size 
        in a grayscale image, bins the values by bin_size, and increments the value at the (min, max) 
        coordinate in separate hash arrays.
        
        :param image: 2D numpy array of grayscale image
        :param segment_size: Number of pixels in each segment (horizontal or vertical)
        :param bin_size: The bin size to reduce the granularity of grayscale values
        :return: 3D numpy array with two stacked hash arrays: one for horizontal and one for vertical segments
        """
        # Calculate the size of the hash arrays based on the bin size
        hash_size = 256 // bin_size
        hash_array_horizontal = np.zeros((hash_size, hash_size), dtype=np.uint16)
        hash_array_vertical = np.zeros((hash_size, hash_size), dtype=np.uint16)
        
        # Process horizontal segments in one go using NumPy's vectorization
        # Reshape each row into chunks of `segment_size` for fast min/max computation
        reshaped_image = image.reshape(image.shape[0], image.shape[1] // segment_size, segment_size)
        min_vals = reshaped_image.min(axis=2) // bin_size  # Compute min values per segment
        max_vals = reshaped_image.max(axis=2) // bin_size  # Compute max values per segment

        # Loop through the min/max values and update the hash array
        for min_val, max_val in zip(min_vals.ravel(), max_vals.ravel()):
            hash_array_horizontal[min_val, max_val] += 1

        # Process vertical segments by transposing the image and repeating the same steps
        reshaped_image_transposed = image.T.reshape(image.shape[1], image.shape[0] // segment_size, segment_size)
        min_vals_vertical = reshaped_image_transposed.min(axis=2) // bin_size  # Compute min values per segment
        max_vals_vertical = reshaped_image_transposed.max(axis=2) // bin_size  # Compute max values per segment

        # Loop through the vertical min/max values and update the hash array
        for min_val, max_val in zip(min_vals_vertical.ravel(), max_vals_vertical.ravel()):
            hash_array_vertical[min_val, max_val] += 1

        # Stack the two hash arrays along a new axis (depth) to create a 3D array
        stacked_hash_arrays = np.stack([hash_array_horizontal, hash_array_vertical], axis=-1)

        return stacked_hash_arrays

    def gradient_blocks_hash(self, grayscale_image: np.ndarray, block_size: int = 8):
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
        height, width, depth = grayscale_image.shape
        block_gradients = np.array(block_gradients).reshape(height // block_size, width // block_size)
        block_angles = np.array(block_angles).reshape(height // block_size, width // block_size)
        
        # return block_gradients, block_angles
        # return block_angles
        # return block_gradients
        return np.stack((block_gradients, block_angles), axis=-1)
    

    
    # def observation_to_privacy_observation(self, observation: np.ndarray, samples=3000) -> np.ndarray:
    #     """
    #     Grayscale test
    #     """

    #     # gray_image = np.dot(observation[...,:3], [0.2989, 0.5870, 0.1140])
    #     # gray_image = np.expand_dims(gray_image.astype(np.uint8), axis=-1)

    #     # print(gray_image.shape)
    #     # return np.expand_dims(observation[:, :, 0], axis=-1)
    #     image = Image.fromarray(observation).convert("L")

    #     g = np.expand_dims(np.array(image), axis=-1) 

    #     # plt.imshow(g, cmap='gray')  # Set color map to 'gray' for grayscale
    #     # plt.axis('off')  # Optional: turn off axis labels and ticks
    #     # plt.show()
        
    #     return g
    
    # def observation_to_privacy_observation(self, observation: np.ndarray, samples=3000) -> np.ndarray:
    #     """
    #     Given a regular observation from the camera, convert it into a privacy preserving image hash
    #     Circle hash function
    #     """
    #     image_hash = np.zeros((256//self.bin_size, 256//self.bin_size, 1), dtype=np.uint8)

    #     # Convert observation to grayscale
    #     gray_image = np.dot(observation[...,:3], [0.2989, 0.5870, 0.1140])
    #     gray_image = gray_image.astype(np.uint8)

    #     for _ in range(samples):
    #         circle_points = self.generate_circular_points(256, 256, min_radius=10, max_radius=10, num_points=100)
    #         pixel_values = np.array([gray_image[y,x] for x,y in circle_points])
    #         min_val = np.min(pixel_values)//self.bin_size
    #         max_val = np.max(pixel_values)//self.bin_size
    #         # Add a cap for uint 8 (for now so we can render grayscale images, it shouldnt be necessary)
    #         image_hash[min_val, max_val] = min(image_hash[min_val, max_val] + 1, 255)

    #     return image_hash
    
    # def observation_to_privacy_observation(self, observation: np.ndarray) -> np.ndarray:
    #     """
    #     Privacy hash function where we get the min and max value in every patch
    #     """
    #     length = 256
    #     patch_size = 8
    #     image_hash = np.zeros((length//self.bin_size, length//self.bin_size, 1), dtype=np.uint16)

    #     # Convert observation to grayscale
    #     gray_image = np.dot(observation[...,:3], [0.2989, 0.5870, 0.1140])
    #     gray_image = gray_image.astype(np.uint8)
    #     # 64 for 256px image
    #     reshaped_array = gray_image.reshape(length//patch_size, patch_size, length//patch_size, patch_size)

    #     min_values = reshaped_array.min(axis=(1, 3)) // self.bin_size
    #     max_values = reshaped_array.max(axis=(1, 3)) // self.bin_size

    #     for min_val, max_val in zip(min_values.ravel(), max_values.ravel()):
    #         image_hash[min_val, max_val] += 1


    #     return image_hash

    def generate_circular_points(self, img_width, img_height, min_radius, max_radius, num_points=100):
        """
        Returns a list of x,y points representing a single random circle on the image
        The number of points depends on num_points which can be adjusted for different resolutions
        """
        radius = random.randint(min_radius, max_radius)
        center_x = random.randint(1 + radius, img_width - radius - 1)
        center_y = random.randint(1 + radius, img_height- radius - 1)

        angles = np.linspace(0, 2*np.pi, num_points)

        # Calculate x and y coorindates for each angle
        x_coords = center_x + radius * np.cos(angles)
        y_coords = center_y + radius * np.sin(angles)

        coordinates = np.column_stack((x_coords, y_coords))

        return np.round(coordinates).astype(int)


    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.quit()
        if hasattr(self, "proc") and self.proc is not None:
            self.proc.quit()

    def set_reward_fn(self, reward_fn: Callable) -> None:
        self.viewer.set_reward_fn(reward_fn)

    def set_episode_over_fn(self, ep_over_fn: Callable) -> None:
        self.viewer.set_episode_over_fn(ep_over_fn)

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        for _ in range(self.frame_skip):
            self.viewer.take_action(action)
            observation, reward, done, info = self.viewer.observe()
        
        if self.is_record:
            self.save_frame(observation, self.frame_number)
        if self.is_privacy:
            observation = self.observation_to_privacy_observation(observation)
            if self.is_record:
                self.save_privacy_frame(observation, self.frame_number)
        self.frame_number += 1

        # Log the x,y positions so we can plot later if needed
        self.x_positions.append(self.viewer.handler.x)
        self.z_positions.append(self.viewer.handler.z)

        # Adding in a fake delay to simulate long processing time (remove later)
        # time.sleep(0.11)

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        # Activate hand brake, so the car does not move
        self.viewer.handler.send_control(0, 0, 1.0)
        time.sleep(0.1)
        self.viewer.reset()
        self.viewer.handler.send_control(0, 0, 1.0)
        time.sleep(0.2)
        observation, reward, done, info = self.viewer.observe()

        # TODO, add privacy preserving logic here
        if self.is_privacy:
            observation = self.observation_to_privacy_observation(observation)

        return observation

    def render(self, mode: str = "human", close: bool = False) -> Optional[np.ndarray]:
        if close:
            self.viewer.quit()

        return self.viewer.render(mode)

    def is_game_over(self) -> bool:
        return self.viewer.is_game_over()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class GeneratedRoadsEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="generated_road", *args, **kwargs)


class WarehouseEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="warehouse", *args, **kwargs)


class AvcSparkfunEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="sparkfun_avc", *args, **kwargs)


class GeneratedTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="generated_track", *args, **kwargs)


class MountainTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="mountain_track", *args, **kwargs)


class RoboRacingLeagueTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="roboracingleague_1", *args, **kwargs)


class WaveshareEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="waveshare", *args, **kwargs)


class MiniMonacoEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="mini_monaco", *args, **kwargs)


class WarrenTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="warren", *args, **kwargs)


class ThunderhillTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="thunderhill", *args, **kwargs)


class CircuitLaunchEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="circuit_launch", *args, **kwargs)
