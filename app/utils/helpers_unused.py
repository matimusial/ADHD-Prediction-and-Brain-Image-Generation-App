import os
import pyedflib
import pandas as pd
import numpy as np
from scipy.io import loadmat

from app.services.data_processing.mri_processing import prepare_x_y


def convert_mat_to_csv_and_edf(input_directory, csv_output_directory, edf_output_directory, edf_channel_labels,
                               sample_rate=128):
    """
    Converts .mat files to both CSV and EDF format.

    Parameters:
    - input_directory: Directory containing .mat files.
    - csv_output_directory: Directory where CSV files will be saved.
    - edf_output_directory: Directory where EDF files will be saved.
    - edf_channel_labels: List of labels for EEG channels.
    - sample_rate: Sample rate for EDF file (default 128 Hz).
    """
    os.makedirs(csv_output_directory, exist_ok=True)
    os.makedirs(edf_output_directory, exist_ok=True)

    mat_files = [f for f in os.listdir(input_directory) if f.endswith('.mat')]

    for mat_file in mat_files:
        try:
            mat_path = os.path.join(input_directory, mat_file)
            mat_data = loadmat(mat_path)

            # Convert all arrays in .mat file to CSV format
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray):
                    df = pd.DataFrame(value)
                    csv_file = os.path.join(csv_output_directory, f'{os.path.splitext(mat_file)[0]}_{key}.csv')
                    df.to_csv(csv_file, index=False)

            eeg_data = mat_data['eeg'].T  # Transpose EEG data to match channel layout
            if eeg_data.shape[0] != len(edf_channel_labels):
                print(f"Error: convert_mat_to_csv_and_edf - Incorrect number of EEG channels in {mat_file}")

            n_channels = len(edf_channel_labels)

            edf_file_path = os.path.join(edf_output_directory, f'{os.path.splitext(mat_file)[0]}.edf')
            with pyedflib.EdfWriter(edf_file_path, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
                # Prepare channel info for EDF file
                channel_info = []
                for i, label in enumerate(edf_channel_labels):
                    ch_dict = {
                        'label': label,
                        'dimension': 'uV',
                        'sample_rate': sample_rate,
                        'physical_min': np.min(eeg_data[i]),
                        'physical_max': np.max(eeg_data[i]),
                        'digital_min': -32768,
                        'digital_max': 32767,
                        'transducer': '',
                        'prefilter': ''
                    }
                    channel_info.append(ch_dict)

                f.setSignalHeaders(channel_info)
                f.writeSamples(eeg_data)
        except Exception as e:
            print(f"Error: convert_mat_to_csv_and_edf - {e}")


def make_predict_data(adhd_data, control_data):
    """
    Prepares data for prediction by randomly selecting samples from ADHD and control groups.

    Parameters:
    - adhd_data: List of ADHD data samples.
    - control_data: List of control data samples.

    Returns:
    - adhd_data_tt: Remaining ADHD data after extraction of prediction samples.
    - control_data_tt: Remaining control data after extraction of prediction samples.
    - x_pred: Combined ADHD and control prediction samples.
    - y_pred: Labels for prediction data (1 for ADHD, 0 for control).
    """
    try:
        # Randomly choose samples for prediction
        adhd_index = np.random.choice(range(0, len(adhd_data)), size=4, replace=False)
        control_index = np.random.choice(range(0, len(control_data)), size=4, replace=False)

        adhd_pred = [adhd_data[i] for i in adhd_index]
        control_pred = [control_data[i] for i in control_index]

        # Remaining data for training/testing
        adhd_data_tt = [adhd_data[i] for i in range(len(adhd_data)) if i not in adhd_index]
        control_data_tt = [control_data[i] for i in range(len(control_data)) if i not in control_index]

        # Create labels for prediction data
        y_adhd_pred = np.ones(len(adhd_pred))
        y_control_pred = np.zeros(len(control_pred))
        y_pred = np.hstack((y_adhd_pred, y_control_pred))
        x_pred = adhd_pred + control_pred

        return adhd_data_tt, control_data_tt, x_pred, y_pred

    except Exception as e:
        print(f"Error: make_pred_data - {e}")


def make_predict_data_mri(adhd_raw, control_raw):
    """
    Prepares MRI data for prediction by selecting random samples from ADHD and control groups.

    Parameters:
    - adhd_raw: List of raw ADHD MRI data.
    - control_raw: List of raw control MRI data.

    Returns:
    - x_pred: MRI data for prediction.
    - y_pred: Labels for prediction data (1 for ADHD, 0 for control).
    - adhd_updated: Remaining ADHD MRI data after extraction of prediction samples.
    - control_updated: Remaining control MRI data after extraction of prediction samples.
    """
    from app.properties.mri_config import CNN_SINGLE_INPUT_SHAPE_MRI

    adhd_pred = []
    adhd_updated = []
    control_pred = []
    control_updated = []

    # Randomly select MRI samples for prediction
    adhd_random_indices = np.random.choice(range(len(adhd_raw)), size=5, replace=False)
    control_random_indices = np.random.choice(range(len(control_raw)), size=5, replace=False)

    for i in range(len(adhd_raw)):
        if i in adhd_random_indices:
            adhd_pred.append(adhd_raw[i])
        else:
            adhd_updated.append(adhd_raw[i])

    for i in range(len(control_raw)):
        if i in control_random_indices:
            control_pred.append(control_raw[i])
        else:
            control_updated.append(control_raw[i])

    x_pred, y_pred = prepare_x_y(adhd_pred, control_pred, CNN_SINGLE_INPUT_SHAPE_MRI)

    return x_pred, y_pred, adhd_updated, control_updated

