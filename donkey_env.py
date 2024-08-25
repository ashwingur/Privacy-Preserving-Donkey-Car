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
    
    def poop(self):
        print("POOP")

    def save_frame(self, observation, id: int):
        # Convert the observation (which is a NumPy array) to an image
        image = Image.fromarray(observation)
        image_path = f"frames/frame_{id:04d}.png"
        image.save(image_path)
        self.images_array.append(image_path)
    
    def save_privacy_frame(self, privacy_observation, id: int):
        # print(privacy_observation)
        # print(np.max(privacy_observation))
        # Convert the observation (which is a NumPy array) to an image
        image = Image.fromarray(np.squeeze(privacy_observation*20), mode="L")
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


    def get_privacy_observation_space(self) -> spaces.Box:
        return spaces.Box(0, 255, (256//self.bin_size, 256//self.bin_size, 1), dtype=np.uint8)
    
    def observation_to_privacy_observation(self, observation: np.ndarray, samples=2000) -> np.ndarray:
        """
        Given a regular observation from the camera, convert it into a privacy preserving image hash
        """
        # TODO implement privacy preserving logic, ensure it matches get_privacy_observation_space
        image_hash = np.zeros((256//self.bin_size, 256//self.bin_size, 1), dtype=np.uint8)

        # Convert observation to grayscale
        gray_image = np.dot(observation[...,:3], [0.2989, 0.5870, 0.1140])
        gray_image = gray_image.astype(np.uint8)

        for _ in range(samples):
            circle_points = self.generate_circular_points(160, 120, min_radius=10, max_radius=10, num_points=100)
            pixel_values = np.array([gray_image[y,x] for x,y in circle_points])
            min_val = np.min(pixel_values)//self.bin_size
            max_val = np.max(pixel_values)//self.bin_size
            # Add a cap for uint 8 (for now so we can render grayscale images, it shouldnt be necessary)
            image_hash[min_val, max_val] = min(image_hash[min_val, max_val] + 1, 255)

        # plt.imshow(image_hash, cmap='gray')
        # plt.imshow(image_hash, cmap='hot', interpolation='nearest')
        # plt.colorbar()  # Add a colorbar to show the intensity scale
        # plt.title('Heatmap')
        # plt.xlabel('Max intensity')
        # plt.ylabel('Min intensity')
        # plt.show()

        return image_hash

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
            self.save_privacy_frame(observation, self.frame_number)
        self.frame_number += 1

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        # Activate hand brake, so the car does not move
        self.viewer.handler.send_control(0, 0, 1.0)
        time.sleep(0.1)
        self.viewer.reset()
        self.viewer.handler.send_control(0, 0, 1.0)
        time.sleep(0.1)
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
