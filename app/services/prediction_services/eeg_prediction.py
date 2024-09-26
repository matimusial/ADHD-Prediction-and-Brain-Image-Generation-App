import ast

import numpy as np

from app.services.data_processing.eeg_processing import filter_bandpass_eeg_data, clip_eeg_extremes, normalize_eeg_data, \
    split_into_frames
from app.services.gui_services.alerts import show_warning_alert


def process_and_predict_eeg(data, model, model_info):
    """
    Processes EEG data and makes predictions using the model.

    :param data: The EEG data to process.
    :param model: The model to use for predictions.
    :param model_info: Information about the model.
    :return: The prediction result.
    """
    import app.properties.cnn_config

    input_shape = ast.literal_eval(model_info[1])
    channels = input_shape[0]
    app.properties.cnn_config.EEG_SIGNAL_FRAME_SIZE = input_shape[1]

    if data.shape[0] != channels:
        return

    try:
        data_filtered = filter_bandpass_eeg_data(data)
        data_clipped = clip_eeg_extremes(data_filtered)
        data_normalized = normalize_eeg_data(data_clipped)
        data_framed = split_into_frames(np.array(data_normalized))
    except Exception as e:
        show_warning_alert(f"Error processing MRI data: {e}")
        print(f"Error: process_and_predict_eeg_data - {e}")

    return model.predict(data_framed)
