import tensorflow as tf
import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

def main(folder_path: str):
    # Prepare a dictionary to store data by step
    data_dict = defaultdict(lambda: {'Wall Time': None, 'Step': None})

    # Iterate through the summary events and extract required data
    for e in tf.compat.v1.train.summary_iterator("log/donkey-warren-track-v0_privacy/4x4_quadrant_minmax_2/events.out.tfevents.1725530113.ashwin.46278.0"):
        wall_time = e.wall_time
        step = e.step
        for v in e.summary.value:
            tag = v.tag
            simple_value = v.simple_value

            # Store data in the dictionary
            data_dict[step]['Wall Time'] = wall_time
            data_dict[step]['Step'] = step
            data_dict[step][tag] = simple_value

    # Extract all unique tags for the headers
    all_tags = set()
    for step_data in data_dict.values():
        all_tags.update(step_data.keys())

    # Convert the set of tags to a sorted list
    all_tags = sorted(all_tags)

    # Define the CSV file path
    csv_file_path = os.path.join(folder_path, 'data.csv')
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Write the data to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=all_tags)
        # Write the header
        writer.writeheader()
        # Write the data rows
        for step, step_data in sorted(data_dict.items()):
            writer.writerow(step_data)

    print(f"Data has been successfully saved to {csv_file_path}")


    # Prepare the figure for subplots
    num_plots = len(all_tags) - 2  # Excluding 'Wall Time' and 'Step'
    num_columns = 2  # Set number of columns to 2
    num_rows = (num_plots + 1) // num_columns  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 4))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    # Index for subplots
    plot_index = 0

    # Generate and save a plot for each unique tag
    for tag in all_tags:
        if tag not in ['Wall Time', 'Step']:
            steps = []
            values = []

            for step_data in data_dict.values():
                if tag in step_data:
                    steps.append(step_data['Step'])
                    values.append(step_data[tag])

            # Save each plot individually
            plt.figure()
            plt.plot(steps, values, label=tag)
            plt.xlabel('Step')
            plt.ylabel('Value')
            # plt.title(f'{tag} over Steps')
            plt.legend()

            # Save the plot as a PNG file
            plot_file_path = os.path.join(folder_path, f"{tag.replace('/', '_')}.png")
            plt.savefig(plot_file_path)
            # Save as an EPS file
            plot_file_path = os.path.join(folder_path, f"{tag.replace('/', '_')}.eps")
            plt.savefig(plot_file_path, format='eps')
            plt.close()

            # Add to subplot for combined image
            axs[plot_index].plot(steps, values, label=tag)
            axs[plot_index].set_xlabel('Step')
            axs[plot_index].set_ylabel('Value')
            axs[plot_index].set_title(f'{tag} over Steps')
            axs[plot_index].legend()

            plot_index += 1

    # Adjust layout and save the combined subplot figure
    plt.tight_layout()
    combined_plot_path = os.path.join(folder_path, 'combined_plot.png')
    fig.savefig(combined_plot_path)
    plt.close(fig)

    print(f"Individual plots and combined subplot have been successfully saved in the folder: {folder_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <folder_path>")
    else:
        folder_path = sys.argv[1]
        main(folder_path)
