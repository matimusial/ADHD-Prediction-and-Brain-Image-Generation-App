import copy
import os

import mne
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split


def read_eeg_raw(path_or_folder):
    """
    Loads raw EEG data from either a .mat file or a folder containing subfolders with EEG data files (CSV/EDF).

    If a .mat file is provided, the function reads and returns the EEG data stored in the file.
    If a folder is provided, the function processes all files in two subfolders ('adhd' and 'control'),
    converts CSV/EDF files into .mat format, and returns the EEG data along with metadata.

    Args:
        path_or_folder (str): The path to a .mat file or a folder containing EEG data in 'adhd' and 'control' subfolders.

    Returns:
        - If the input is a .mat file, returns the EEG data stored in the file.
        - If the input is a folder, returns a tuple containing:
            - adhd_data (list): List of EEG data arrays from the 'adhd' subfolder.
            - control_data (list): List of EEG data arrays from the 'control' subfolder.
            - table_structures (list): List of dictionaries, each containing the file path and shape of the loaded data.
            - adhd_files_count (int): Number of files processed from the 'adhd' subfolder.
            - control_files_count (int): Number of files processed from the 'control' subfolder.
    """

    adhd_files_count = 0
    control_files_count = 0
    table_structures = []

    if os.path.isfile(path_or_folder) and path_or_folder.endswith('.mat'):
        # Load and return data from a single .mat file
        mat_data = loadmat(path_or_folder, mat_dtype=True)
        file, _ = os.path.splitext(os.path.basename(path_or_folder))
        if file not in mat_data:
            print(f"Key {file} not found in the .mat file.")

        return mat_data[file].T

    elif os.path.isdir(path_or_folder):
        # Handle folder with 'adhd' and 'control' subfolders
        subfolders = ["adhd", "control"]
        adhd_data = []
        control_data = []

        for subfolder in subfolders:
            current_folder = os.path.join(path_or_folder, subfolder)
            if not os.path.isdir(current_folder):
                print(f"Subfolder {current_folder} does not exist.")

            for file in os.listdir(current_folder):
                if file.endswith('.csv') or file.endswith('.edf'):
                    csv_or_edf_file = os.path.join(current_folder, file)
                    mat_file_name = os.path.splitext(file)[0] + '.mat'

                    if file.endswith('.csv'):
                        df = pd.read_csv(csv_or_edf_file)
                        data_matrix = df.values
                        key_name = os.path.splitext(file)[0]
                        savemat(os.path.join(current_folder, mat_file_name), {key_name: data_matrix})

                    elif file.endswith('.edf'):
                        raw = mne.io.read_raw_edf(csv_or_edf_file, preload=True)
                        data = raw.get_data()
                        key_name = os.path.splitext(file)[0]
                        savemat(os.path.join(current_folder, mat_file_name), {key_name: data.transpose()})

        for subfolder in subfolders:
            current_folder = os.path.join(path_or_folder, subfolder)
            mat_files = [f for f in os.listdir(current_folder) if f.endswith('.mat')]

            for mat_file in mat_files:
                file_path = os.path.join(current_folder, mat_file)
                loaded_data = loadmat(file_path, mat_dtype=True)
                file_name, _ = os.path.splitext(mat_file)
                if file_name not in loaded_data:
                    print(f"Key {file_name} not found in .mat file {mat_file}.")

                data = loaded_data[file_name].T
                table_structure = {
                    'file': file_path,
                    'shape': data.shape,
                }
                table_structures.append(table_structure)

                if "adhd" in subfolder:
                    adhd_data.append(loaded_data[file_name].T)
                    adhd_files_count += 1
                elif "control" in subfolder:
                    control_data.append(loaded_data[file_name].T)
                    control_files_count += 1

        return adhd_data, control_data, table_structures, adhd_files_count, control_files_count

    else:
        print("Error: read_eeg_raw - path is not a valid .mat file or folder.")


def filter_bandpass_eeg_data(adhd_data, control_data=None, band_type=2):
    """
    Applies a bandpass filter to EEG data.

    Args:
    adhd_data (list): List of EEG data for the ADHD group.
    control_data (list, optional): List of EEG data for the control group. Required for version 1.
    band_type (int): Type of frequency band to filter. Default is 2.

    Returns:
    list: Bandpass filtered EEG data or tuple of two lists if control_data is provided.
    """
    from app.properties.cnn_config import CUTOFFS, FS
    order = 4
    cutoff = CUTOFFS[band_type]
    low_cutoff = cutoff[0]
    high_cutoff = cutoff[1]
    b, a = signal.butter(order, [low_cutoff / (0.5 * FS), high_cutoff / (0.5 * FS)], btype='bandpass')

    adhd_filtered = [signal.filtfilt(b, a, data) for data in adhd_data]

    if control_data is not None:
        control_filtered = [signal.filtfilt(b, a, data) for data in control_data]
        return adhd_filtered, control_filtered

    return adhd_filtered


def normalize_eeg_data(adhd_data, control_data=None):
    """

    Args:
    adhd_data (list): List of EEG data for the ADHD group.
    control_data (list, optional): List of EEG data for the control group.

    Returns:
    list: Normalized EEG data or tuple of two lists if control_data is provided.
    """
    def scale_data_to_range(data):
        from app.properties.cnn_config import CNN_INPUT_SHAPE
        data_scaled = copy.deepcopy(data)
        for i, patient_data in enumerate(data):
            for j in range(CNN_INPUT_SHAPE[0]):
                min_value = np.min(patient_data[j]).astype(np.float64)
                max_value = np.max(patient_data[j]).astype(np.float64)
                if max_value != min_value:
                    data_scaled[i][j] = 2 * ((patient_data[j] - min_value) / (max_value - min_value)) - 1
                else:
                    data_scaled[i][j] = np.zeros_like(patient_data[j])
        return data_scaled

    if control_data is not None:
        if len(adhd_data) <= 1 or len(control_data) <= 1:
            print("Error: normalize_eeg_data - This function requires more than one patient in each group")

        adhd_normalized = scale_data_to_range(adhd_data)
        control_normalized = scale_data_to_range(control_data)
        return adhd_normalized, control_normalized

    if len(adhd_data) <= 1:
        print("Error: normalize_eeg_data - This function requires more than one patient")

    return scale_data_to_range(adhd_data)


def clip_eeg_extremes(adhd_data, control_data=None):
    """
    Clips extreme values in EEG data to a specified percentile for both ADHD and control groups.

    Args:
        adhd_data (list): List of EEG data for the ADHD group.
        control_data (list, optional): List of EEG data for the control group.

    Returns:
        list: Clipped EEG data or tuple of two lists if control_data is provided.
    """

    def clip_extreme_values(data):
        from app.properties.cnn_config import CNN_INPUT_SHAPE
        percentile = 99.8
        data_clipped = copy.deepcopy(data)
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile
        for i, patient_data in enumerate(data):
            for j in range(CNN_INPUT_SHAPE[0]):
                channel_data = patient_data[j]
                lower_threshold = np.percentile(channel_data, lower_percentile)
                upper_threshold = np.percentile(channel_data, upper_percentile)
                data_clipped[i][j] = np.clip(channel_data, a_min=lower_threshold, a_max=upper_threshold)
        return data_clipped

    if control_data is not None:
        if len(adhd_data) <= 1 or len(control_data) <= 1:
            print("Error: clip_eeg_extremes - This function requires more than one patient in each group")

        adhd_clipped = clip_extreme_values(adhd_data)
        control_clipped = clip_extreme_values(control_data)
        return adhd_clipped, control_clipped

    if len(adhd_data) <= 1:
        print("Error: clip_eeg_extremes - This function requires more than one patient")

    return clip_extreme_values(adhd_data)


def split_into_frames(data):
    """Splits the data into frames of EEG_SIGNAL_FRAME_SIZE.

    Args:
        data (numpy.ndarray): The eeg data to be split.

    Returns:
        numpy.ndarray: The framed eeg data.
    """
    from app.properties.cnn_config import EEG_SIGNAL_FRAME_SIZE
    try:
        if data.shape[1] < EEG_SIGNAL_FRAME_SIZE:
            print("Error: split_into_frames - The number of samples is less than the frame size.")
        num_frames = data.shape[1] // EEG_SIGNAL_FRAME_SIZE
        framed_data = np.zeros((num_frames, data.shape[0], EEG_SIGNAL_FRAME_SIZE))

        for i in range(num_frames):
            framed_data[i, :, :] = data[:, i * EEG_SIGNAL_FRAME_SIZE: (i + 1) * EEG_SIGNAL_FRAME_SIZE]

        return framed_data
    except Exception as e:
        print(f"Error: split_into_frames - {e}")


def prepare_for_cnn(adhd_data, control_data):
    """Prepares EEG data for training and testing CNN.

    Args:
        adhd_data (list or np.ndarray): The EEG data for ADHD patients.
        control_data (list or np.ndarray): The EEG data for control (non-ADHD) patients.

    Returns:
        tuple: A tuple containing the following:
            - x_train (np.ndarray): Training data for CNN.
            - y_train (np.ndarray): Labels for training data (1 for ADHD, 0 for control).
            - x_test (np.ndarray): Testing data for CNN.
            - y_test (np.ndarray): Labels for testing data (1 for ADHD, 0 for control).
    """
    from app.properties.cnn_config import TEST_SIZE_EEG_CNN
    try:
        adhd_framed_list = [split_into_frames(patient_data) for patient_data in adhd_data]
        control_framed_list = [split_into_frames(patient_data) for patient_data in control_data]

        adhd_framed = np.concatenate(adhd_framed_list, axis=0)
        control_framed = np.concatenate(control_framed_list, axis=0)

        y_adhd = np.ones(adhd_framed.shape[0])
        y_control = np.zeros(control_framed.shape[0])

        x = np.concatenate((control_framed, adhd_framed))
        y = np.concatenate((np.array(y_control), np.array(y_adhd)))

        x_4d = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))
        x_train, x_test, y_train, y_test = train_test_split(x_4d, y, test_size=TEST_SIZE_EEG_CNN, shuffle=True)

        return x_train, y_train, x_test, y_test

    except Exception as e:
        print(f"Error: prepare_for_cnn - {e}")
