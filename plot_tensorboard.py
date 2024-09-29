import tensorflow as tf
import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def main(file_name_tuples: list, folder_path: str):
    '''
    :param file_name_tuples: List of tuples with display names and file paths
    :param folder_path: Folder to save the plots to
    '''
    
    # Prepare a dictionary to store data by step for each file
    data_dicts = []
    
    for display_name, file_path in file_name_tuples:
        # Prepare a dictionary for this file's data
        data_dict = defaultdict(lambda: {'Wall Time': None, 'Step': None})
        
        # Iterate through the summary events and extract required data
        for e in tf.compat.v1.train.summary_iterator(file_path):
            wall_time = e.wall_time
            step = e.step
            for v in e.summary.value:
                tag = v.tag
                simple_value = v.simple_value

                # Store data in the dictionary
                data_dict[step]['Wall Time'] = wall_time
                data_dict[step]['Step'] = step
                data_dict[step][tag] = simple_value

        data_dicts.append((data_dict, display_name))

    # Extract all unique tags for the headers across all files
    all_tags = set()
    for data_dict, _ in data_dicts:
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
        # Write the data rows for each file
        for data_dict, display_name in data_dicts:
            for step, step_data in sorted(data_dict.items()):
                writer.writerow(step_data)

    print(f"Data has been successfully saved to {csv_file_path}")

    # Generate and save individual plots for each unique tag
    for tag in all_tags:
        if tag not in ['Wall Time', 'Step']:
            plt.figure(figsize=(10, 6))
            
            # Plot the data for each file
            for data_dict, display_name in data_dicts:
                steps = []
                values = []
                for step_data in data_dict.values():
                    if tag in step_data:
                        steps.append(step_data['Step'])
                        values.append(step_data[tag])

                plt.plot(steps, values, label=display_name)
            
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.title(f'{tag} over Steps')
            plt.legend()

            # Save each plot individually as PNG and EPS
            plot_file_name = tag.replace('/', '_')
            png_path = os.path.join(folder_path, f"{plot_file_name}.png")
            eps_path = os.path.join(folder_path, f"{plot_file_name}.eps")

            plt.savefig(png_path)
            plt.savefig(eps_path, format='eps')
            plt.close()

            print(f"Plot for '{tag}' saved as PNG and EPS.")

    print(f"All individual plots have been successfully saved in the folder: {folder_path}")


if __name__ == "__main__":
    # Hardcoded list of tuples with display names and file paths
    file_name_tuples = [
        # ("Run 1", "log/donkey-warren-track-v0_privacy/4x4_quadrant_minmax_2/events.out.tfevents.1725530113.ashwin.46278.0"),
        ("Run 2", "log/donkey-warren-track-v0_privacy/greyscale_5/events.out.tfevents.1726719515.ashwin.28795.0"),
        ("Patch Hash", "log/donkey-warren-track-v0_privacy/patch_hash_8x8_3/events.out.tfevents.1726731714.ashwin.38097.0")
    ]
    
    folder_path = "plots"
    main(file_name_tuples, folder_path)
