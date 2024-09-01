import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.cluster import KMeans
import os
import datetime
import shutil

time = r"\2024-06-29_12-55-48"


# Manager functions
def create_analysis_folder(base_directory='D:/PhD thesis/Data Analysis Code-Python'):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    analysis_folder_path = os.path.join(base_directory, f'Analysis_{timestamp}')
    os.makedirs(analysis_folder_path, exist_ok=True)
    return analysis_folder_path


def copy_input_files(input_files, destination_folder):
    for file_path in input_files:
        shutil.copy(file_path, destination_folder)


def save_plot_as_pdf(figure, filename, output_folder):
    file_path = os.path.join(output_folder, f'{filename}.pdf')
    figure.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.close(figure)


# Order Data
def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    step_data = []

    for line in lines:
        parts = line.split()

        if len(parts) == 1 and parts[0].isdigit():
            # This line indicates the start of a new step
            if step_data:
                data.append(step_data)
            step_data = []

        elif len(parts) >= 3:
            try:
                x, y, z = map(float, parts[:3])
                step_data.append((x, y, z))
            except ValueError:
                continue

    if step_data:
        data.append(step_data)

    return data


def calculate_step_sizes(data):
    step_sizes = []
    num_steps = len(data)
    num_Agents = len(data[0])

    for step in range(1, num_steps):
        for Agent in range(num_Agents):
            prev_position = np.array(data[step - 1][Agent][1:4])
            current_position = np.array(data[step][Agent][1:4])
            step_size = np.linalg.norm(current_position - prev_position)
            step_sizes.append(step_size)

    return step_sizes


# Calculate step sizes for a specific Agent
def calculate_step_sizes_for_Agent(data, Agent_id):
    step_sizes = []
    num_steps = len(data)

    for step in range(1, num_steps):
        prev_position = None
        current_position = None

        for Agent in data[step - 1]:
            if Agent[0] == Agent_id:
                prev_position = np.array(Agent[1:4])
                break

        for Agent in data[step]:
            if Agent[0] == Agent_id:
                current_position = np.array(Agent[1:4])
                break

        if prev_position is not None and current_position is not None:
            step_size = np.linalg.norm(current_position - prev_position)
            step_sizes.append(step_size)

    return step_sizes


# Analysis Data
def plot_histogram(step_sizes, title, output_folder):
    fig, ax = plt.subplots()
    ax.hist(step_sizes, bins=30, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Step Size')
    ax.set_ylabel('Frequency')
    save_plot_as_pdf(fig, title.replace(' ', '_').lower(), output_folder)


def plot_histogram_for_time_step(data, time_step, output_folder):
    step_sizes = []
    if time_step < 1 or time_step >= len(data):
        raise ValueError("Invalid time step. Please select a time step between 1 and the total number of steps.")
    num_Agents = len(data[0])
    for Agent in range(num_Agents):
        prev_position = np.array(data[time_step - 1][Agent][1:4])
        current_position = np.array(data[time_step][Agent][1:4])
        step_size = np.linalg.norm(current_position - prev_position)
        step_sizes.append(step_size)
    plot_histogram(step_sizes, f'Histogram of Step Sizes at Time Step {time_step}', output_folder)


def plot_histogram_for_Agent(data, Agent_id, output_folder):
    step_sizes = calculate_step_sizes_for_Agent(data, Agent_id)
    plot_histogram(step_sizes, f'Histogram of Step Sizes for Agent {Agent_id} over All Time Steps', output_folder)


def plot_density_map(density_map, grid_size, output_folder):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    non_zero_indices = np.argwhere(density_map > 0)
    densities = density_map[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]]
    sizes = densities * 100
    ax.scatter(non_zero_indices[:, 0] * grid_size, non_zero_indices[:, 1] * grid_size,
               non_zero_indices[:, 2] * grid_size, s=sizes, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Density Map of Particles')
    save_plot_as_pdf(fig, 'density_map', output_folder)


def plot_trajectory(data, output_folder, Agent_id=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num_steps = len(data)
    num_Agents = len(data[0])
    if Agent_id is not None:
        trajectory = np.array([step[Agent_id] for step in data])
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=f'Agent {Agent_id} Trajectory')
    else:
        for Agent in range(num_Agents):
            trajectory = np.array([step[Agent] for step in data])
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=f'Agent {Agent} Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Trajectory of Agents Over Time')
    plt.legend()
    save_plot_as_pdf(fig, 'trajectory_plot' if Agent_id is None else f'trajectory_Agent_{Agent_id}', output_folder)


def calculate_density(data, grid_size):
    # Check the structure of data
    all_coords = []
    for step in data:
        for Agent in step:
            if len(Agent) == 3:  # Ensure the Agent tuple has 3 elements
                all_coords.append((Agent[0], Agent[1], Agent[2]))
            else:
                print(f"Unexpected Agent structure: {Agent}")

    all_coords = np.array(all_coords)

    if all_coords.size == 0:
        raise ValueError("No valid coordinates found in data.")

    # Extent of the coordinate data
    min_coord = np.floor(all_coords.min(axis=0) / grid_size) * grid_size
    max_coord = np.ceil(all_coords.max(axis=0) / grid_size) * grid_size
    dims = ((max_coord - min_coord) / grid_size).astype(int) + 1

    # Create an empty density map
    density_map = np.zeros(dims)

    # Populate the density map
    for step in data:
        for Agent in step:
            if len(Agent) == 3:  # Ensure the Agent tuple has 3 elements
                indices = ((np.array(Agent) - min_coord) / grid_size).astype(int)
                density_map[tuple(indices)] += 1

    return density_map


def calculate_displacement_statistics(data):
    num_steps = len(data)
    num_Agents = len(data[0])

    displacements = {Agent: [] for Agent in range(num_Agents)}
    speeds = {Agent: [] for Agent in range(num_Agents)}

    for Agent in range(num_Agents):
        total_displacement = 0.0
        for step in range(1, num_steps):
            prev_position = np.array(data[step - 1][Agent])
            current_position = np.array(data[step][Agent])
            displacement = np.linalg.norm(current_position - prev_position)
            displacements[Agent].append(displacement)
            speeds[Agent].append(displacement)  # Assuming unit time between steps
            total_displacement += displacement

        displacements[Agent] = np.array(displacements[Agent])
        speeds[Agent] = np.array(speeds[Agent])

    statistics = {
        Agent: {
            "mean_displacement": np.mean(displacements[Agent]),
            "variance_displacement": np.var(displacements[Agent]),
            "mean_speed": np.mean(speeds[Agent]),
            "total_displacement": np.sum(displacements[Agent])
        }
        for Agent in range(num_Agents)
    }

    return statistics


def calculate_correlation(data1, data2):
    num_steps = min(len(data1), len(data2))  # Ensure we only compare up to the shortest dataset
    num_Agents = min(len(data1[0]), len(data2[0]))  # Ensure the number of Agents is consistent

    correlations = {}

    for Agent in range(num_Agents):
        displacements1 = []
        displacements2 = []

        for step in range(1, num_steps):
            displacement1 = np.linalg.norm(np.array(data1[step][Agent]) - np.array(data1[step - 1][Agent]))
            displacement2 = np.linalg.norm(np.array(data2[step][Agent]) - np.array(data2[step - 1][Agent]))

            displacements1.append(displacement1)
            displacements2.append(displacement2)

        # Calculate Pearson correlation coefficient
        if len(displacements1) > 1:  # Correlation requires at least two data points
            correlation, _ = pearsonr(displacements1, displacements2)
            correlations[Agent] = correlation
        else:
            correlations[Agent] = None  # Not enough data to calculate correlation

    return correlations


def calculate_correlation_matrix(data):
    num_steps = len(data)
    num_Agents = len(data[0])

    # Initialize a matrix to hold the displacements
    displacements = np.zeros((num_Agents, num_steps - 1))

    # Calculate displacements for each Agent across all time steps
    for Agent in range(num_Agents):
        for step in range(1, num_steps):
            displacement = np.linalg.norm(np.array(data[step][Agent]) - np.array(data[step - 1][Agent]))
            displacements[Agent, step - 1] = displacement

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(displacements)

    return correlation_matrix


def plot_heatmap_and_save(correlation_matrix, output_file):
    plt.figure(figsize=(20, 16))  # Increase the figure size for better visibility
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True,
                annot_kws={"size": 8})  # Adjust font size for better readability
    plt.title('Correlation Heatmap of Agent Displacements')
    plt.xlabel(f'{da1} Index')
    plt.ylabel(f'{da2} Index')

    plt.savefig(output_file, format='pdf', bbox_inches='tight')  # Save as PDF with tight bounding box
    plt.close()  # Close the figure to avoid display


def calculate_msd(data):
    num_steps = len(data)
    num_Agents = len(data[0])

    msd = np.zeros(num_steps)

    for step in range(num_steps):
        sum_displacement = 0.0
        for Agent in range(num_Agents):
            initial_position = np.array(data[0][Agent])
            current_position = np.array(data[step][Agent])
            displacement = np.linalg.norm(current_position - initial_position) ** 2
            sum_displacement += displacement
        msd[step] = sum_displacement / num_Agents

    return msd


def cluster_particles(data, num_clusters=3):
    final_positions = np.array([data[-1][Agent] for Agent in range(len(data[0]))])
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(final_positions)
    labels = kmeans.labels_

    return labels


def calculate_angular_distribution(data):
    num_steps = len(data)
    num_Agents = len(data[0])

    angles = []

    for Agent in range(num_Agents):
        for step in range(1, num_steps - 1):
            vec1 = np.array(data[step][Agent]) - np.array(data[step - 1][Agent])
            vec2 = np.array(data[step + 1][Agent]) - np.array(data[step][Agent])
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                angles.append(np.degrees(angle))

    return angles


def write_summary_file(destination_folder, content):
    summary_file_path = os.path.join(destination_folder, 'analysis_summary.txt')
    with open(summary_file_path, 'w') as file:
        file.write(content)


def main_analysis(input_files, base_directory):
    analysis_folder = create_analysis_folder(base_directory)
    copy_input_files(input_files, analysis_folder)

    for file_path in input_files:
        data = read_xyz(file_path)
        filename = os.path.basename(file_path).replace('.xyz', '')
        print("file_path")
        step_sizes = calculate_step_sizes(data)
        plot_histogram(step_sizes, f'Histogram of Step Sizes for {filename}', analysis_folder)

        grid_size = 10
        density_map = calculate_density(data, grid_size)
        plot_density_map(density_map, grid_size, analysis_folder)

        plot_trajectory(data, analysis_folder)

        displacement_stats = calculate_displacement_statistics(data)

        if len(input_files) > 1:
            data2 = read_xyz(input_files[1]) if file_path == input_files[0] else read_xyz(input_files[0])
            correlations = calculate_correlation(data, data2)

        correlation_matrix = calculate_correlation_matrix(data)
        plot_heatmap_and_save(correlation_matrix, os.path.join(analysis_folder, f'{filename}_correlation_heatmap.pdf'))

    summary_content = "This folder contains the outputs of the analysis performed on the provided data files."
    write_summary_file(analysis_folder, summary_content)

    print(f"All files have been organized and saved in: {analysis_folder}")


file_path = r"D:\PhD thesis\SimulationSV\cmake-build-debug\builds"
file1 = r"\Vesicles_movement.xyz"
file2 = r"\SynapsinI_movement.xyz"
da1 = "Vesicle"
da2 = "Synapsin"
file_path1 = file_path + time + file1
file_path2: str = file_path + time + file2

input_files = [
    file_path1, file_path2
]
main_analysis(input_files, r'D:/PhD thesis/Data Analysis Code-Python')

main_analysis(input_files, r'D:/PhD thesis/Data Analysis Code-Python')
