"""
BUILT UPON THE TEMPLATE FROM:
    file: ppo_train.py
    author: Tawn Kramer
    date: 13 October 2018
    notes: ppo2 test from stable-baselines here:
    https://github.com/hill-a/stable-baselines


Got warren track working with the following params:
    "max_cte": 10,
    "steer_limit": 0.5,
    "throttle_min": 0.1,
    "throttle_max": 0.5,
    60000 timesteps (could likely reduce this)
"""
import argparse
import uuid
import os
import shutil

import gym
from stable_baselines3 import PPO
import gym_donkeycar
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio
import cv2

def show_observation(obs):
    plt.imshow(obs)
    plt.axis('off')  # Hide the axes for a cleaner image display
    plt.show()

def save_frame(observation, id: int, images_array):
    # Convert the observation (which is a NumPy array) to an image
    image = Image.fromarray(observation)
    image_path = f"frames/frame_{id:04d}.png"
    image.save(image_path)
    images_array.append(image_path)


def generate_mp4_video(images_array, video_name='output_video.mp4', fps=15, scale_factor=2):
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

    # Determine if the image is grayscale or RGB
    if len(first_image.shape) == 2:  # Grayscale image
        height, width = first_image.shape
        layers = 1
    else:  # RGB image
        height, width, layers = first_image.shape

    new_height, new_width = height * scale_factor, width * scale_factor

    # Initialize video writer with color enabled (3-channel output)
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height), isColor=True)

    for image_file in images_array:
        image = imageio.imread(image_file)
        if len(image.shape) == 2:  # Grayscale
            # Upscale grayscale image
            upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            # Convert grayscale to 3-channel by replicating the single channel
            upscaled_image = cv2.cvtColor(upscaled_image, cv2.COLOR_GRAY2RGB)
        else:
            # Upscale RGB image
            upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            # Convert RGB to BGR for OpenCV
            upscaled_image = cv2.cvtColor(upscaled_image, cv2.COLOR_RGB2BGR)

        video.write(upscaled_image)

    video.release()
    print(f"Video saved as {video_name}")



if __name__ == "__main__":
    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-mountain-track-v0",
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
    ]

    parser = argparse.ArgumentParser(description="ppo_train")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--test", action="store_true", help="load the trained model and play")
    parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
    parser.add_argument("--privacy", action="store_true", help="enable the privacy preserving filter")
    parser.add_argument("--record", action="store_true", help="enable recording of frames")
    parser.add_argument(
        "--env_name", type=str, default="donkey-mountain-track-v0", help="name of donkey sim environment", choices=env_list
    )
    parser.add_argument(
        "--log_name",
        type=str,
        help="custom name for the TensorBoard logging run"
    )
    args = parser.parse_args()

    # Check if --log_name is required
    if not args.test and not args.log_name:
        parser.error("--log_name is required for training. If you are testing add the --test flag.")

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)

    env_id = f"{args.env_name}{'_privacy' if args.privacy else ''}"

    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256


    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "very fast car",
        "font_size": 50,
        "racer_name": "PPO",
        "country": "USA",
        "bio": "Learning to drive w PPO RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
        "steer_limit": 0.5,
        "throttle_min": 0.2,
        "throttle_max": 0.3,
        "privacy": args.privacy, # Indicate whether privacy hashing is enabled
        "record": args.record, # If we should be recording the frames (to view the observations)
        # Modify the camera resolution
        "cam_config": {
            "img_w": IMAGE_WIDTH,
            "img_h": IMAGE_HEIGHT
        },
        "cam_resolution": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    }

    if args.test:
        # Make an environment test our trained policy
        env = gym.make(args.env_name, conf=conf)

        model = PPO.load(env_id)
        images = []
        image_folder = 'frames/'
        # Clear the directory if it exists
        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)
        os.makedirs(image_folder, exist_ok=True)

        obs = env.reset()
        for i in range(500):
            # Display the observation
            # show_observation(obs)

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # save_frame(observation=obs, id=i, images_array=images )
            env.render()
            if done:
                obs = env.reset()
        # save_video(f"{env_id}-vid.mp4", images_array=images)
        # print(images)
        # generate_mp4_video(images, f"{env_id}-vid.mp4")
        if args.record:
            images = sorted([f"frames/{f}" for f in os.listdir("frames") if os.path.isfile(os.path.join("frames", f))])
            generate_mp4_video(images, f"{env_id}_real_image-vid.mp4")
            if args.privacy:
                images = sorted([f"privacy_frames/{f}" for f in os.listdir("privacy_frames") if os.path.isfile(os.path.join("privacy_frames", f))])
                generate_mp4_video(images, f"{env_id}-vid.mp4")

        print("done testing")

    else:
        # make gym env
        env = gym.make(args.env_name, conf=conf)

        # create cnn policy
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f"./log/{env_id}/", n_steps=2048)

        # set up model in learning mode with goal number of timesteps to complete
        try:
            model.learn(total_timesteps=55000, tb_log_name=args.log_name)
        except KeyboardInterrupt:
            if args.record:
                images = sorted([f"frames/{f}" for f in os.listdir("frames") if os.path.isfile(os.path.join("frames", f))])
                generate_mp4_video(images, f"{env_id}_real_image-vid.mp4")
                images = sorted([f"privacy_frames/{f}" for f in os.listdir("privacy_frames") if os.path.isfile(os.path.join("privacy_frames", f))])
                generate_mp4_video(images, f"{env_id}-vid.mp4")
            env.close()
            exit()

        obs = env.reset()

        # We are not training in this loop, just testing
        for i in range(200):
            action, _states = model.predict(obs, deterministic=True)
            # env.poop()

            obs, reward, done, info = env.step(action)

            try:
                env.render()
            except Exception as e:
                print(e)
                print("failure in render, continuing...")

            if done:
                obs = env.reset()

            if i % 100 == 0:
                print("saving...")
                model.save(env_id)

        # Save the agent
        model.save(env_id)
        print("done training")

    env.plot_xy(f"xy/{env_id}_{'test' if args.test else 'train'}.png")

    env.close()