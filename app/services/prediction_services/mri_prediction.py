import numpy as np

from app.services.data_processing.mri_processing import trim_mri, normalize_mri
from app.services.gui_services.alerts import show_warning_alert


def process_and_predict_mri(data, model):
    """
    Processes MRI data and makes predictions using the model.

    :param data: The MRI data to process.
    :param model: The model to use for predictions.
    :return: The prediction result.
    """
    from app.properties.mri_config import CNN_INPUT_SHAPE_MRI

    try:
        data_trimmed = np.reshape(trim_mri(data, CNN_INPUT_SHAPE_MRI[:2]), CNN_INPUT_SHAPE_MRI)
        data_normalized = normalize_mri(data_trimmed)
        img_for_predict = data_normalized.reshape(1, data_normalized.shape[0], data_normalized.shape[1], 1)

    except Exception as e:
        show_warning_alert(f"Error processing MRI data: {e}")
        print(f"Error: process_and_predict_mri - {e}")
        return None

    return model.predict(img_for_predict)
